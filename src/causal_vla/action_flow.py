"""Source-free steering in a VLA action expert's denoising flow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

FlowRoute = Literal["state", "representation"]
RepresentationCondition = Literal["joint", "vision", "language", "null"]


def flatten_flow_directions(
    directions: tuple[Tensor, ...],
) -> tuple[Tensor, tuple[tuple[int, ...], ...]]:
    """Flatten per-call direction batches without losing their tensor shapes."""

    if not directions:
        raise ValueError("At least one flow direction call is required")
    sample_count = directions[0].shape[0]
    if sample_count == 0:
        raise ValueError("Flow directions require at least one training sample")
    shapes: list[tuple[int, ...]] = []
    flattened: list[Tensor] = []
    for direction in directions:
        if direction.shape[0] != sample_count:
            raise ValueError("All flow calls must have the same sample count")
        if direction.ndim < 2 or not bool(torch.isfinite(direction).all()):
            raise ValueError("Flow directions must be finite batched tensors")
        shapes.append(tuple(direction.shape[1:]))
        flattened.append(direction.detach().double().reshape(sample_count, -1))
    return torch.cat(flattened, dim=1), tuple(shapes)


def unflatten_flow_direction(
    flattened: Tensor,
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[Tensor, ...]:
    """Restore one flattened prediction to its per-denoising-call tensors."""

    if flattened.ndim != 1:
        raise ValueError("A flattened flow prediction must be one-dimensional")
    offset = 0
    restored: list[Tensor] = []
    for shape in shapes:
        size = int(np.prod(shape))
        restored.append(flattened[offset : offset + size].reshape(shape).float())
        offset += size
    if offset != flattened.numel():
        raise ValueError("Flow shapes do not consume the full flattened prediction")
    return tuple(restored)


@dataclass(frozen=True)
class RidgeFlowRegressor:
    """Dual ridge regressor from an internal VLM state to action-flow corrections."""

    mean: Tensor
    scale: Tensor
    target_mean: Tensor
    normalized_keys: Tensor
    dual_weights: Tensor
    alpha: float

    @classmethod
    def fit(cls, keys: Tensor, targets: Tensor, *, alpha: float) -> RidgeFlowRegressor:
        """Fit a deterministic float64 dual ridge model on CPU."""

        if keys.shape[0] != targets.shape[0] or keys.shape[0] == 0:
            raise ValueError("Keys and targets require the same nonzero sample count")
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("Ridge alpha must be positive and finite")
        x = keys.detach().double().cpu().reshape(keys.shape[0], -1)
        y = targets.detach().double().cpu().reshape(targets.shape[0], -1)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise ValueError("Ridge training data must be finite")
        mean = x.mean(dim=0, keepdim=True)
        scale = x.std(dim=0, correction=0, keepdim=True).clamp_min(1e-4)
        normalized = ((x - mean) / scale).clamp(-20.0, 20.0)
        target_mean = y.mean(dim=0, keepdim=True)
        centered_targets = y - target_mean
        kernel = normalized @ normalized.transpose(0, 1) / normalized.shape[1]
        regularized = kernel + alpha * torch.eye(len(normalized), dtype=torch.float64)
        dual = torch.linalg.solve(regularized, centered_targets)
        return cls(mean, scale, target_mean, normalized, dual, float(alpha))

    def predict(self, key: Tensor) -> Tensor:
        """Predict one flattened action-flow correction."""

        query = key.detach().cpu().double().reshape(1, -1)
        if query.shape[1] != self.mean.shape[1]:
            raise ValueError("Representation key width does not match the fitted model")
        normalized = ((query - self.mean) / self.scale).clamp(-20.0, 20.0)
        kernel = normalized @ self.normalized_keys.transpose(0, 1)
        kernel = kernel / self.normalized_keys.shape[1]
        prediction = self.target_mean + kernel @ self.dual_weights
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("Ridge controller produced a non-finite correction")
        return prediction[0]


@dataclass(frozen=True)
class StateFlowIndex:
    """Local phase controller indexed by robot proprioception."""

    center: Tensor
    scale: Tensor
    normalized_features: Tensor
    targets: Tensor
    neighbors: int

    @classmethod
    def fit(
        cls,
        features: Tensor,
        targets: Tensor,
        *,
        neighbors: int = 3,
        scale_floor: Tensor | None = None,
    ) -> StateFlowIndex:
        """Fit an inverse-distance KNN action-flow index."""

        if features.ndim != 2 or features.shape[0] != targets.shape[0]:
            raise ValueError("State features must align with flow targets")
        if not 1 <= neighbors <= features.shape[0]:
            raise ValueError("neighbors must be between one and the sample count")
        x = features.detach().double().cpu()
        y = targets.detach().double().cpu()
        center = x.mean(dim=0)
        scale = x.std(dim=0, correction=0)
        if scale_floor is not None:
            floor = scale_floor.detach().double().cpu()
            if floor.shape != scale.shape or bool((floor <= 0).any()):
                raise ValueError("State scale floor must be positive and match feature width")
            scale = torch.maximum(scale, floor)
        else:
            scale = scale.clamp_min(1e-4)
        normalized = (x - center) / scale
        return cls(center, scale, normalized, y, neighbors)

    def predict(self, feature: Tensor) -> tuple[Tensor, float]:
        """Return a flattened local correction and nearest-state distance."""

        query = feature.detach().double().cpu().reshape(-1)
        if query.shape != self.center.shape:
            raise ValueError("State feature width does not match the fitted index")
        distances = torch.linalg.vector_norm(
            self.normalized_features - (query - self.center) / self.scale,
            dim=1,
        )
        values, indices = torch.topk(distances, k=self.neighbors, largest=False)
        weights = values.clamp_min(1e-4).reciprocal()
        weights = weights / weights.sum()
        prediction = torch.sum(weights[:, None] * self.targets[indices], dim=0)
        return prediction, float(values[0])


@dataclass
class MonotoneFlowGate:
    """Route local-to-general once; never re-enter an unsafe local phase."""

    state_distance_threshold: float
    committed_to_representation: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.state_distance_threshold < 0 or not np.isfinite(self.state_distance_threshold):
            raise ValueError("State distance threshold must be nonnegative and finite")

    def route(self, state_distance: float) -> FlowRoute:
        """Choose a controller and latch permanently after the first general route."""

        if not np.isfinite(state_distance) or state_distance < 0:
            raise ValueError("State distance must be nonnegative and finite")
        if state_distance > self.state_distance_threshold:
            self.committed_to_representation = True
        return "representation" if self.committed_to_representation else "state"

    def reset(self) -> None:
        """Start a new rollout before any route has been committed."""

        self.committed_to_representation = False


class ActionFlowAdd:
    """Add one cached correction to each action-head denoising call."""

    def __init__(self, directions: tuple[Tensor, ...], *, scale: float = 1.0):
        if not directions:
            raise ValueError("At least one action-flow direction is required")
        if not np.isfinite(scale):
            raise ValueError("Action-flow scale must be finite")
        self.directions = tuple(direction.detach() for direction in directions)
        self.scale = float(scale)
        self.call_index = 0

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: Tensor,
    ) -> Tensor:
        if self.call_index >= len(self.directions):
            raise RuntimeError("Action head was called more times than fitted directions")
        direction = self.directions[self.call_index].to(output.device, output.dtype)
        self.call_index += 1
        if direction.shape != output.shape:
            raise ValueError(
                f"Action-flow shape mismatch: direction={direction.shape}, output={output.shape}"
            )
        return output + self.scale * direction

    def assert_consumed(self) -> None:
        """Require exactly one correction per fitted denoising call."""

        if self.call_index != len(self.directions):
            raise RuntimeError(
                f"Consumed {self.call_index} of {len(self.directions)} action-flow calls"
            )


@dataclass(frozen=True)
class ActionFlowArtifact:
    """Pickle-free training set and locked controller metadata."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    resolution: int
    fit_base_prompt: str
    fit_target_prompt: str
    runtime_base_prompt: str
    runtime_target_prompt: str
    train_episodes: tuple[int, ...]
    episode_ids: Tensor
    steps: Tensor
    representation_layer: int
    representation_position: int
    representation_keys: Tensor
    state_features: Tensor
    flow_directions: tuple[Tensor, ...]
    ridge_alpha: float
    state_distance_threshold: float
    state_neighbors: int
    noise_seed: int
    simulator_seed: int
    sample_every: int
    vision_keys: Tensor | None = None
    language_keys: Tensor | None = None
    null_keys: Tensor | None = None
    neutral_prompt: str = ""
    image_ablation_value: float = 0.5

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported action-flow schema {self.schema_version}")
        sample_count = len(self.episode_ids)
        if sample_count == 0 or len(self.steps) != sample_count:
            raise ValueError("Action-flow artifact requires aligned episode and step arrays")
        if self.representation_keys.shape[0] != sample_count:
            raise ValueError("Representation keys do not align with sample count")
        if self.state_features.ndim != 2 or self.state_features.shape[0] != sample_count:
            raise ValueError("State features do not align with sample count")
        if not self.flow_directions or any(
            direction.shape[0] != sample_count for direction in self.flow_directions
        ):
            raise ValueError("Every flow call must align with the sample count")
        if self.fit_base_prompt == self.fit_target_prompt:
            raise ValueError("Fitting prompts must differ")
        if self.runtime_base_prompt == self.runtime_target_prompt:
            raise ValueError("Runtime prompts must differ")
        if self.sample_every <= 0:
            raise ValueError("sample_every must be positive")
        if not 1 <= self.state_neighbors <= sample_count:
            raise ValueError("state_neighbors must be between one and the sample count")
        if self.ridge_alpha <= 0 or not np.isfinite(self.ridge_alpha):
            raise ValueError("ridge_alpha must be positive and finite")
        if self.state_distance_threshold < 0 or not np.isfinite(self.state_distance_threshold):
            raise ValueError("state_distance_threshold must be nonnegative and finite")
        for tensor in (self.representation_keys, self.state_features):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Action-flow artifact tensors must be finite")
        conditioned = (self.vision_keys, self.language_keys, self.null_keys)
        if self.schema_version == 2 and any(tensor is None for tensor in conditioned):
            raise ValueError("Schema 2 requires vision, language, and null keys")
        for conditioned_tensor in conditioned:
            if conditioned_tensor is None:
                continue
            if conditioned_tensor.shape != self.representation_keys.shape:
                raise ValueError("Conditioned representation keys must share one shape")
            if not bool(torch.isfinite(conditioned_tensor).all()):
                raise ValueError("Conditioned representation keys must be finite")
        if not 0.0 <= self.image_ablation_value <= 1.0:
            raise ValueError("image_ablation_value must be in [0, 1]")
        flatten_flow_directions(self.flow_directions)

    def keys_for(self, condition: RepresentationCondition) -> Tensor:
        """Return the training keys for one preregistered conditioning arm."""

        if condition == "joint":
            return self.representation_keys
        key = {
            "vision": self.vision_keys,
            "language": self.language_keys,
            "null": self.null_keys,
        }[condition]
        if key is None:
            raise ValueError(
                f"Artifact schema {self.schema_version} has no {condition!r} conditioning keys"
            )
        return key

    def metadata(self) -> dict[str, object]:
        """Return auditable metadata without tensor payloads."""

        first_shape = list(self.flow_directions[0].shape[1:])
        return {
            "schema_version": self.schema_version,
            "checkpoint": self.checkpoint,
            "suite": self.suite,
            "task_id": self.task_id,
            "resolution": self.resolution,
            "fit_base_prompt": self.fit_base_prompt,
            "fit_target_prompt": self.fit_target_prompt,
            "runtime_base_prompt": self.runtime_base_prompt,
            "runtime_target_prompt": self.runtime_target_prompt,
            "cross_template": (
                self.fit_base_prompt != self.runtime_base_prompt
                or self.fit_target_prompt != self.runtime_target_prompt
            ),
            "train_episodes": list(self.train_episodes),
            "representation_layer": self.representation_layer,
            "representation_position": self.representation_position,
            "ridge_alpha": self.ridge_alpha,
            "state_distance_threshold": self.state_distance_threshold,
            "state_neighbors": self.state_neighbors,
            "noise_seed": self.noise_seed,
            "simulator_seed": self.simulator_seed,
            "sample_every": self.sample_every,
            "sample_count": len(self.episode_ids),
            "flow_calls": len(self.flow_directions),
            "flow_tensor_shape": first_shape,
            "intervention_pathway": "action_out_proj",
            "language_token_intervention": False,
            "conditioning_modes": (
                ["joint", "vision", "language", "null"] if self.schema_version >= 2 else ["joint"]
            ),
            "neutral_prompt": self.neutral_prompt,
            "image_ablation_value": self.image_ablation_value,
        }

    def save(self, path: str | Path) -> Path:
        """Atomically serialize metadata and tensors to an NPZ file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "metadata_json": np.frombuffer(
                json.dumps(self.metadata(), sort_keys=True).encode("utf-8"), dtype=np.uint8
            ),
            "episode_ids": self.episode_ids.detach().cpu().numpy(),
            "steps": self.steps.detach().cpu().numpy(),
            "representation_keys": self.representation_keys.detach().float().cpu().numpy(),
            "state_features": self.state_features.detach().float().cpu().numpy(),
        }
        optional_keys = {
            "vision_keys": self.vision_keys,
            "language_keys": self.language_keys,
            "null_keys": self.null_keys,
        }
        for name, tensor in optional_keys.items():
            if tensor is not None:
                arrays[name] = tensor.detach().float().cpu().numpy()
        for index, direction in enumerate(self.flow_directions):
            arrays[f"flow_call_{index}"] = direction.detach().float().cpu().numpy()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **cast(dict[str, Any], arrays))
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ActionFlowArtifact:
        """Load an action-flow artifact without enabling pickle."""

        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata_json"].tobytes().decode("utf-8"))
            calls = tuple(
                torch.from_numpy(
                    np.array(payload[f"flow_call_{index}"], dtype=np.float32, copy=True)
                )
                for index in range(int(metadata["flow_calls"]))
            )
            return cls(
                schema_version=int(metadata["schema_version"]),
                checkpoint=str(metadata["checkpoint"]),
                suite=str(metadata["suite"]),
                task_id=int(metadata["task_id"]),
                resolution=int(metadata["resolution"]),
                fit_base_prompt=str(metadata["fit_base_prompt"]),
                fit_target_prompt=str(metadata["fit_target_prompt"]),
                runtime_base_prompt=str(metadata["runtime_base_prompt"]),
                runtime_target_prompt=str(metadata["runtime_target_prompt"]),
                train_episodes=tuple(int(value) for value in metadata["train_episodes"]),
                episode_ids=torch.from_numpy(np.array(payload["episode_ids"], copy=True)),
                steps=torch.from_numpy(np.array(payload["steps"], copy=True)),
                representation_layer=int(metadata["representation_layer"]),
                representation_position=int(metadata["representation_position"]),
                representation_keys=torch.from_numpy(
                    np.array(payload["representation_keys"], dtype=np.float32, copy=True)
                ),
                state_features=torch.from_numpy(
                    np.array(payload["state_features"], dtype=np.float32, copy=True)
                ),
                flow_directions=calls,
                ridge_alpha=float(metadata["ridge_alpha"]),
                state_distance_threshold=float(metadata["state_distance_threshold"]),
                state_neighbors=int(metadata["state_neighbors"]),
                noise_seed=int(metadata["noise_seed"]),
                simulator_seed=int(metadata["simulator_seed"]),
                sample_every=int(metadata["sample_every"]),
                vision_keys=(
                    torch.from_numpy(np.array(payload["vision_keys"], dtype=np.float32, copy=True))
                    if "vision_keys" in payload
                    else None
                ),
                language_keys=(
                    torch.from_numpy(
                        np.array(payload["language_keys"], dtype=np.float32, copy=True)
                    )
                    if "language_keys" in payload
                    else None
                ),
                null_keys=(
                    torch.from_numpy(np.array(payload["null_keys"], dtype=np.float32, copy=True))
                    if "null_keys" in payload
                    else None
                ),
                neutral_prompt=str(metadata.get("neutral_prompt", "")),
                image_ablation_value=float(metadata.get("image_ablation_value", 0.5)),
            )
