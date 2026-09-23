"""Fit and serialize source-free residual steering directions."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.residual_runtime import (
    capture_pre_and_post_layer_residuals,
    vlm_layers,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch

DirectionComponent = Literal["post_residual", "layer_update"]


@dataclass(frozen=True)
class CachedSteeringArtifact:
    """A signed mean residual direction learned entirely from training episodes."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    resolution: int
    base_label: str
    target_label: str
    base_prompt: str
    target_prompt: str
    proposition: str
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    train_episodes: tuple[int, ...]
    noise_seed: int
    simulator_seed: int
    directions: dict[int, tuple[Tensor, ...]]
    fit_base_prompt: str | None = None
    fit_target_prompt: str | None = None
    fit_positions: tuple[int, ...] | None = None
    fit_base_token_ids: tuple[int, ...] = ()
    fit_target_token_ids: tuple[int, ...] = ()
    runtime_base_token_ids: tuple[int, ...] = ()
    runtime_target_token_ids: tuple[int, ...] = ()
    fit_base_span_token_ids: tuple[int, ...] = ()
    fit_target_span_token_ids: tuple[int, ...] = ()
    runtime_base_span_token_ids: tuple[int, ...] = ()
    runtime_target_span_token_ids: tuple[int, ...] = ()
    direction_component: DirectionComponent = "post_residual"
    input_directions: dict[int, tuple[Tensor, ...]] = field(default_factory=dict)
    runtime_changed_token_positions: tuple[int, ...] = ()
    intervention_overlaps_changed_tokens: bool = True

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError(f"Unsupported steering schema {self.schema_version}")
        if self.direction_component not in {"post_residual", "layer_update"}:
            raise ValueError(f"Unsupported direction component {self.direction_component!r}")
        if self.fit_base_prompt is None:
            object.__setattr__(self, "fit_base_prompt", self.base_prompt)
        if self.fit_target_prompt is None:
            object.__setattr__(self, "fit_target_prompt", self.target_prompt)
        if self.fit_positions is None:
            object.__setattr__(self, "fit_positions", self.positions)
        if self.base_label == self.target_label:
            raise ValueError("Base and target labels must be distinct")
        if self.base_prompt == self.target_prompt:
            raise ValueError("Runtime base and target prompts must be distinct")
        if self.fit_base_prompt == self.fit_target_prompt:
            raise ValueError("Fit base and target prompts must be distinct")
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("Layers must be nonempty, unique, and sorted")
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise ValueError("Runtime positions must be nonempty and unique")
        if not self.fit_positions or len(set(self.fit_positions)) != len(self.fit_positions):
            raise ValueError("Fit positions must be nonempty and unique")
        if len(self.fit_positions) != len(self.positions) and len(self.fit_positions) != 1:
            raise ValueError("Fit and runtime spans must match unless one fit token is broadcast")
        if set(self.directions) != set(self.layers):
            raise ValueError("Every selected layer must have cached directions")
        if self.input_directions and set(self.input_directions) != set(self.layers):
            raise ValueError("Every selected layer must have cached input directions")
        for collection in (self.directions, self.input_directions):
            for layer, calls in collection.items():
                if not calls:
                    raise ValueError(f"Layer {layer} has no cached direction calls")
                for direction in calls:
                    if direction.ndim < 3:
                        raise ValueError("Directions must have [batch, positions, hidden] shape")
                    if direction.shape[-2] != len(self.fit_positions):
                        raise ValueError("Direction position dimension does not match fit metadata")
                    if not bool(torch.isfinite(direction).all()):
                        raise ValueError("Directions must contain only finite values")

    def metadata(self) -> dict[str, object]:
        """Return JSON metadata without embedding tensor values."""

        direction_keys = {
            str(layer): [f"layer_{layer}_call_{index}" for index in range(len(calls))]
            for layer, calls in self.directions.items()
        }
        input_direction_keys = {
            str(layer): [f"input_layer_{layer}_call_{index}" for index in range(len(calls))]
            for layer, calls in self.input_directions.items()
        }
        return {
            "schema_version": self.schema_version,
            "checkpoint": self.checkpoint,
            "suite": self.suite,
            "task_id": self.task_id,
            "resolution": self.resolution,
            "base_label": self.base_label,
            "target_label": self.target_label,
            "base_prompt": self.base_prompt,
            "target_prompt": self.target_prompt,
            "fit_base_prompt": self.fit_base_prompt,
            "fit_target_prompt": self.fit_target_prompt,
            "proposition": self.proposition,
            "layers": list(self.layers),
            "positions": list(self.positions),
            "fit_positions": list(cast(tuple[int, ...], self.fit_positions)),
            "fit_base_token_ids": list(self.fit_base_token_ids),
            "fit_target_token_ids": list(self.fit_target_token_ids),
            "runtime_base_token_ids": list(self.runtime_base_token_ids),
            "runtime_target_token_ids": list(self.runtime_target_token_ids),
            "fit_base_span_token_ids": list(self.fit_base_span_token_ids),
            "fit_target_span_token_ids": list(self.fit_target_span_token_ids),
            "runtime_base_span_token_ids": list(self.runtime_base_span_token_ids),
            "runtime_target_span_token_ids": list(self.runtime_target_span_token_ids),
            "direction_component": self.direction_component,
            "runtime_changed_token_positions": list(self.runtime_changed_token_positions),
            "intervention_overlaps_changed_tokens": (self.intervention_overlaps_changed_tokens),
            "cross_template": (
                self.fit_base_prompt != self.base_prompt
                or self.fit_target_prompt != self.target_prompt
            ),
            "train_episodes": list(self.train_episodes),
            "noise_seed": self.noise_seed,
            "simulator_seed": self.simulator_seed,
            "direction_keys": direction_keys,
            "input_direction_keys": input_direction_keys,
        }

    def save(self, path: str | Path) -> Path:
        """Atomically save metadata and tensors in a pickle-free NPZ file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for layer, calls in self.directions.items():
            for index, direction in enumerate(calls):
                arrays[f"layer_{layer}_call_{index}"] = direction.detach().float().cpu().numpy()
        for layer, calls in self.input_directions.items():
            for index, direction in enumerate(calls):
                arrays[f"input_layer_{layer}_call_{index}"] = (
                    direction.detach().float().cpu().numpy()
                )
        metadata_bytes = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        arrays["metadata_json"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **cast(dict[str, Any], arrays))
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CachedSteeringArtifact:
        """Load and validate a pickle-free cached steering artifact."""

        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata_json"].tobytes().decode("utf-8"))
            direction_keys = metadata.pop("direction_keys")
            input_direction_keys = metadata.pop("input_direction_keys", {})
            directions = {
                int(layer): tuple(
                    torch.from_numpy(np.array(payload[key], dtype=np.float32, copy=True))
                    for key in keys
                )
                for layer, keys in direction_keys.items()
            }
            input_directions = {
                int(layer): tuple(
                    torch.from_numpy(np.array(payload[key], dtype=np.float32, copy=True))
                    for key in keys
                )
                for layer, keys in input_direction_keys.items()
            }
        return cls(
            schema_version=int(metadata["schema_version"]),
            checkpoint=str(metadata["checkpoint"]),
            suite=str(metadata["suite"]),
            task_id=int(metadata["task_id"]),
            resolution=int(metadata["resolution"]),
            base_label=str(metadata["base_label"]),
            target_label=str(metadata["target_label"]),
            base_prompt=str(metadata["base_prompt"]),
            target_prompt=str(metadata["target_prompt"]),
            proposition=str(metadata["proposition"]),
            layers=tuple(int(layer) for layer in metadata["layers"]),
            positions=tuple(int(position) for position in metadata["positions"]),
            train_episodes=tuple(int(episode) for episode in metadata["train_episodes"]),
            noise_seed=int(metadata["noise_seed"]),
            simulator_seed=int(metadata["simulator_seed"]),
            directions=directions,
            fit_base_prompt=str(metadata.get("fit_base_prompt", metadata["base_prompt"])),
            fit_target_prompt=str(metadata.get("fit_target_prompt", metadata["target_prompt"])),
            fit_positions=tuple(
                int(position) for position in metadata.get("fit_positions", metadata["positions"])
            ),
            fit_base_token_ids=tuple(
                int(token) for token in metadata.get("fit_base_token_ids", [])
            ),
            fit_target_token_ids=tuple(
                int(token) for token in metadata.get("fit_target_token_ids", [])
            ),
            runtime_base_token_ids=tuple(
                int(token) for token in metadata.get("runtime_base_token_ids", [])
            ),
            runtime_target_token_ids=tuple(
                int(token) for token in metadata.get("runtime_target_token_ids", [])
            ),
            fit_base_span_token_ids=tuple(
                int(token) for token in metadata.get("fit_base_span_token_ids", [])
            ),
            fit_target_span_token_ids=tuple(
                int(token) for token in metadata.get("fit_target_span_token_ids", [])
            ),
            runtime_base_span_token_ids=tuple(
                int(token) for token in metadata.get("runtime_base_span_token_ids", [])
            ),
            runtime_target_span_token_ids=tuple(
                int(token) for token in metadata.get("runtime_target_span_token_ids", [])
            ),
            direction_component=cast(
                DirectionComponent, metadata.get("direction_component", "post_residual")
            ),
            input_directions=input_directions,
            runtime_changed_token_positions=tuple(
                int(position) for position in metadata.get("runtime_changed_token_positions", [])
            ),
            intervention_overlaps_changed_tokens=bool(
                metadata.get("intervention_overlaps_changed_tokens", True)
            ),
        )


@dataclass(frozen=True)
class SteeringFitReport:
    """Provenance and training-set stability for a cached steering artifact."""

    schema_version: int
    artifact_path: str
    artifact_sha256: str
    metadata: dict[str, object]
    direction_l2_by_layer: dict[str, tuple[float, ...]]
    mean_cosine_to_direction_by_layer: dict[str, tuple[float, ...]]
    input_direction_l2_by_layer: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable fit report."""

        payload = asdict(self)
        payload["direction_l2_by_layer"] = {
            layer: list(values) for layer, values in self.direction_l2_by_layer.items()
        }
        payload["mean_cosine_to_direction_by_layer"] = {
            layer: list(values) for layer, values in self.mean_cosine_to_direction_by_layer.items()
        }
        payload["input_direction_l2_by_layer"] = {
            layer: list(values) for layer, values in self.input_direction_l2_by_layer.items()
        }
        return payload


def file_sha256(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(first: Tensor, second: Tensor) -> float:
    left = first.float().reshape(-1)
    right = second.float().reshape(-1)
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return 0.0
    return float(torch.dot(left, right).div(denominator).item())


def attended_language_token_ids(batch: dict[str, Any]) -> tuple[int, ...]:
    """Extract the unpadded language token IDs emitted by the real processor."""

    tokens = batch.get("observation.language.tokens")
    mask = batch.get("observation.language.attention_mask")
    if not isinstance(tokens, Tensor) or not isinstance(mask, Tensor):
        raise ValueError("Processed batch has no language token IDs and attention mask")
    flat_tokens = tokens.detach().cpu().reshape(-1)
    flat_mask = mask.detach().cpu().reshape(-1).bool()
    if flat_tokens.shape != flat_mask.shape:
        raise ValueError("Language tokens and attention mask have different shapes")
    return tuple(int(token) for token in flat_tokens[flat_mask].tolist())


def residual_span_token_ids(
    token_ids: tuple[int, ...],
    residual_positions: tuple[int, ...],
    *,
    require_language: bool = True,
) -> tuple[int, ...]:
    """Map SmolVLA residual positions to text IDs; one state token follows text."""

    indices = tuple(len(token_ids) + 1 + position for position in residual_positions)
    invalid = tuple(index for index in indices if index < 0 or index >= len(token_ids))
    if invalid and require_language:
        raise ValueError(
            "A selected residual position does not address a language token; SmolVLA "
            "places one state token after the language sequence"
        )
    return tuple(token_ids[index] for index in indices if 0 <= index < len(token_ids))


def changed_language_residual_positions(
    base_token_ids: tuple[int, ...], target_token_ids: tuple[int, ...]
) -> tuple[int, ...]:
    """Return residual indices of runtime language tokens changed by the prompt pair."""

    if len(base_token_ids) != len(target_token_ids):
        raise ValueError("Runtime base and target prompts must have equal token counts")
    sequence_length_with_state = len(base_token_ids) + 1
    return tuple(
        index - sequence_length_with_state
        for index, (base, target) in enumerate(zip(base_token_ids, target_token_ids, strict=True))
        if base != target
    )


def fit_cached_steering(
    checkpoint: str | Path,
    output: str | Path,
    *,
    base_label: str,
    target_label: str,
    base_prompt: str,
    target_prompt: str,
    proposition: str,
    layers: tuple[int, ...],
    positions: tuple[int, ...],
    train_episodes: tuple[int, ...],
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    simulator_seed: int = 42,
    runtime_base_prompt: str | None = None,
    runtime_target_prompt: str | None = None,
    runtime_positions: tuple[int, ...] | None = None,
    direction_component: DirectionComponent = "post_residual",
) -> SteeringFitReport:
    """Fit a mean target-minus-base residual direction on training observations."""

    if not train_episodes:
        raise ValueError("At least one training episode is required")
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("Layers must be nonempty, unique, and sorted")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Fit positions must be nonempty and unique")
    runtime_base_prompt = runtime_base_prompt or base_prompt
    runtime_target_prompt = runtime_target_prompt or target_prompt
    runtime_positions = runtime_positions or positions
    if not runtime_positions or len(set(runtime_positions)) != len(runtime_positions):
        raise ValueError("Runtime positions must be nonempty and unique")
    if len(runtime_positions) != len(positions) and len(positions) != 1:
        raise ValueError("Fit and runtime spans must match unless one fit token is broadcast")
    if direction_component not in {"post_residual", "layer_update"}:
        raise ValueError(f"Unsupported direction component {direction_component!r}")
    random.seed(noise_seed)
    np.random.seed(noise_seed % (2**32))
    torch.manual_seed(noise_seed)
    activate_action_atlas(action_atlas_root, state_dir)
    adapters = importlib.import_module("experiments.model_adapters")
    adapter = adapters.SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    decoder_layers, final_norm = vlm_layers(policy)

    samples: dict[int, list[list[Tensor]]] = {layer: [] for layer in layers}
    input_samples: dict[int, list[list[Tensor]]] = {layer: [] for layer in layers}
    expected_call_counts: dict[int, int] = {}
    token_ids: dict[str, tuple[int, ...]] = {}
    torch_device = torch.device(device)
    for episode_index in train_episodes:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            if native_prompt != runtime_target_prompt:
                raise ValueError(
                    "Runtime target prompt must exactly match the environment's native instruction"
                )
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            noise = fixed_noise(policy, noise_seed + episode_index, torch_device)
            base_batch = prepare_libero_batch(adapter, observation, base_prompt)
            target_batch = prepare_libero_batch(adapter, observation, target_prompt)
            runtime_base_batch = prepare_libero_batch(adapter, observation, runtime_base_prompt)
            runtime_target_batch = prepare_libero_batch(adapter, observation, runtime_target_prompt)
            observed_token_ids = {
                "fit_base": attended_language_token_ids(base_batch),
                "fit_target": attended_language_token_ids(target_batch),
                "runtime_base": attended_language_token_ids(runtime_base_batch),
                "runtime_target": attended_language_token_ids(runtime_target_batch),
            }
            if token_ids and observed_token_ids != token_ids:
                raise ValueError("Prompt tokenization changed across training episodes")
            token_ids = observed_token_ids
            _, base_inputs, base_residuals = capture_pre_and_post_layer_residuals(
                policy,
                base_batch,
                noise,
                layers=decoder_layers,
                final_norm=final_norm,
                layer_indices=layers,
            )
            _, target_inputs, target_residuals = capture_pre_and_post_layer_residuals(
                policy,
                target_batch,
                noise,
                layers=decoder_layers,
                final_norm=final_norm,
                layer_indices=layers,
            )
        finally:
            environment.close()
        for layer in layers:
            base_calls = base_residuals[layer]
            target_calls = target_residuals[layer]
            if len(base_calls) != len(target_calls):
                raise ValueError(f"Layer {layer} has different base and target call counts")
            expected = expected_call_counts.setdefault(layer, len(base_calls))
            if len(base_calls) != expected:
                raise ValueError(f"Layer {layer} call count changed across training episodes")
            episode_samples: list[Tensor] = []
            episode_input_samples: list[Tensor] = []
            for base, target in zip(base_calls, target_calls, strict=True):
                if base.shape != target.shape:
                    raise ValueError(
                        "Source-free fitting requires token-aligned prompts with equal "
                        f"residual shapes; got {base.shape} and {target.shape}"
                    )
                indices = list(positions)
                difference = target[..., indices, :].float() - base[..., indices, :].float()
                base_input = base_inputs[layer][len(episode_samples)]
                target_input = target_inputs[layer][len(episode_samples)]
                input_difference = (
                    target_input[..., indices, :].float() - base_input[..., indices, :].float()
                )
                if direction_component == "layer_update":
                    difference = difference - input_difference
                episode_samples.append(difference.cpu())
                episode_input_samples.append(input_difference.cpu())
            samples[layer].append(episode_samples)
            input_samples[layer].append(episode_input_samples)

    directions: dict[int, tuple[Tensor, ...]] = {}
    input_directions: dict[int, tuple[Tensor, ...]] = {}
    norms: dict[str, tuple[float, ...]] = {}
    input_norms: dict[str, tuple[float, ...]] = {}
    cosines: dict[str, tuple[float, ...]] = {}
    for layer in layers:
        call_directions: list[Tensor] = []
        call_norms: list[float] = []
        call_input_directions: list[Tensor] = []
        call_input_norms: list[float] = []
        call_cosines: list[float] = []
        for call_index in range(expected_call_counts[layer]):
            call_samples = [episode[call_index] for episode in samples[layer]]
            direction = torch.stack(call_samples).mean(dim=0)
            call_directions.append(direction)
            call_norms.append(float(direction.norm().item()))
            call_cosines.append(
                float(np.mean([_cosine(sample, direction) for sample in call_samples]))
            )
            input_call_samples = [episode[call_index] for episode in input_samples[layer]]
            input_direction = torch.stack(input_call_samples).mean(dim=0)
            call_input_directions.append(input_direction)
            call_input_norms.append(float(input_direction.norm().item()))
        directions[layer] = tuple(call_directions)
        input_directions[layer] = tuple(call_input_directions)
        norms[str(layer)] = tuple(call_norms)
        input_norms[str(layer)] = tuple(call_input_norms)
        cosines[str(layer)] = tuple(call_cosines)

    changed_runtime_positions = changed_language_residual_positions(
        token_ids["runtime_base"], token_ids["runtime_target"]
    )
    artifact = CachedSteeringArtifact(
        schema_version=2,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        resolution=resolution,
        base_label=base_label,
        target_label=target_label,
        base_prompt=runtime_base_prompt,
        target_prompt=runtime_target_prompt,
        proposition=proposition,
        layers=layers,
        positions=runtime_positions,
        train_episodes=train_episodes,
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        directions=directions,
        fit_base_prompt=base_prompt,
        fit_target_prompt=target_prompt,
        fit_positions=positions,
        fit_base_token_ids=token_ids["fit_base"],
        fit_target_token_ids=token_ids["fit_target"],
        runtime_base_token_ids=token_ids["runtime_base"],
        runtime_target_token_ids=token_ids["runtime_target"],
        fit_base_span_token_ids=residual_span_token_ids(
            token_ids["fit_base"], positions, require_language=False
        ),
        fit_target_span_token_ids=residual_span_token_ids(
            token_ids["fit_target"], positions, require_language=False
        ),
        runtime_base_span_token_ids=residual_span_token_ids(
            token_ids["runtime_base"], runtime_positions, require_language=False
        ),
        runtime_target_span_token_ids=residual_span_token_ids(
            token_ids["runtime_target"], runtime_positions, require_language=False
        ),
        direction_component=direction_component,
        input_directions=input_directions,
        runtime_changed_token_positions=changed_runtime_positions,
        intervention_overlaps_changed_tokens=bool(
            set(runtime_positions) & set(changed_runtime_positions)
        ),
    )
    artifact_path = artifact.save(output).resolve()
    return SteeringFitReport(
        schema_version=2,
        artifact_path=str(artifact_path),
        artifact_sha256=file_sha256(artifact_path),
        metadata=artifact.metadata(),
        direction_l2_by_layer=norms,
        mean_cosine_to_direction_by_layer=cosines,
        input_direction_l2_by_layer=input_norms,
    )


def matched_random_directions(
    artifact: CachedSteeringArtifact,
    *,
    seed: int,
) -> dict[int, tuple[Tensor, ...]]:
    """Generate deterministic random directions matched per token to learned norms."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result: dict[int, tuple[Tensor, ...]] = {}
    for layer, calls in artifact.directions.items():
        random_calls: list[Tensor] = []
        for direction in calls:
            random_direction = torch.randn(
                direction.shape,
                generator=generator,
                dtype=torch.float32,
            )
            learned_norm = direction.float().norm(dim=-1, keepdim=True)
            random_norm = random_direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            random_calls.append(random_direction * learned_norm / random_norm)
        result[layer] = tuple(random_calls)
    return result
