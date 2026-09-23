"""Trajectory-conditioned KNN steering at a causally selected expert residual."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import torch
from torch import Tensor

from causal_vla.causal_trace_runtime import PrefixSnapshot, prefix_snapshot
from causal_vla.expert_steering_runtime import (
    ExpertSteeringSpec,
    capture_expert_output,
)
from causal_vla.expert_trace_runtime import (
    capture_expert_post_residuals,
    expert_flow_states,
    expert_layers,
)
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.object_intervention import (
    apply_hidden_object_scene,
    object_conflict_task,
)
from causal_vla.residual_runtime import decode_action
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch


class KNNCrossValidationRow(TypedDict):
    neighbors: int
    temporal_candidates: int | None
    scale: float
    loo_target_mse_recovery: float
    loo_target_cosine: float


@dataclass(frozen=True)
class ExpertPrototypeBank:
    """Per-flow-step conflict features and correct-minus-conflict directions."""

    spec: ExpertSteeringSpec
    num_steps: int
    features: dict[int, Tensor]
    directions: dict[int, Tensor]
    episode_indices: dict[int, Tensor]
    environment_steps: dict[int, Tensor]
    neighbors: int
    scale: float
    temporal_candidates: int | None = None

    def __post_init__(self) -> None:
        expected = set(range(self.num_steps))
        if self.num_steps <= 0 or set(self.features) != expected:
            raise ValueError("Prototype features must cover every flow step")
        if set(self.directions) != expected or set(self.episode_indices) != expected:
            raise ValueError("Prototype targets and episodes must cover every flow step")
        if set(self.environment_steps) != expected:
            raise ValueError("Prototype environment steps must cover every flow step")
        if self.neighbors <= 0 or not math.isfinite(self.scale):
            raise ValueError("Prototype K and scale must be valid")
        if self.temporal_candidates is not None and (self.temporal_candidates < self.neighbors):
            raise ValueError("Temporal candidate count must be at least K")
        for step in range(self.num_steps):
            features = self.features[step]
            directions = self.directions[step]
            episodes = self.episode_indices[step]
            environment_steps = self.environment_steps[step]
            if features.ndim != 2 or directions.ndim != 2:
                raise ValueError("Prototype features and directions must be matrices")
            if features.shape != directions.shape:
                raise ValueError("Prototype feature and direction shapes must match")
            if (
                features.shape[0] != episodes.numel()
                or features.shape[0] != environment_steps.numel()
            ):
                raise ValueError("Prototype metadata length must match the sample count")
            if self.neighbors > features.shape[0]:
                raise ValueError("Prototype K exceeds the number of samples")


def _expert_prototype_payload(bank: ExpertPrototypeBank) -> dict[str, object]:
    """Return the tensor-only payload shared by standalone and composite banks."""

    return {
        "schema_version": 2,
        "controller_type": "knn",
        "spec": asdict(bank.spec),
        "num_steps": bank.num_steps,
        "neighbors": bank.neighbors,
        "scale": bank.scale,
        "temporal_candidates": bank.temporal_candidates,
        "features": bank.features,
        "directions": bank.directions,
        "episode_indices": bank.episode_indices,
        "environment_steps": bank.environment_steps,
    }


def save_expert_prototype_bank(bank: ExpertPrototypeBank, path: str | Path) -> None:
    """Serialize a tensor-only prototype bank for source-free evaluation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _expert_prototype_payload(bank)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _expert_prototype_from_payload(payload: object) -> ExpertPrototypeBank:
    """Validate and construct a prototype bank from a tensor-only payload."""

    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported expert prototype bank schema")
    spec_payload = payload.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("Prototype bank is missing its steering spec")
    spec = ExpertSteeringSpec(
        name=str(spec_payload["name"]),
        layer=int(spec_payload["layer"]),
        token_positions=tuple(int(value) for value in spec_payload["token_positions"]),
        interface=cast(Literal["post_residual", "expert_output"], spec_payload["interface"]),
    )
    def tensor_map(name: str) -> dict[int, Tensor]:
        value = payload.get(name)
        if not isinstance(value, dict) or not all(
            isinstance(key, int) and isinstance(tensor, Tensor) for key, tensor in value.items()
        ):
            raise ValueError(f"Prototype bank field {name!r} is malformed")
        return cast(dict[int, Tensor], value)

    return ExpertPrototypeBank(
        spec=spec,
        num_steps=int(payload["num_steps"]),
        features=tensor_map("features"),
        directions=tensor_map("directions"),
        episode_indices=tensor_map("episode_indices"),
        environment_steps=tensor_map("environment_steps"),
        neighbors=int(payload["neighbors"]),
        scale=float(payload["scale"]),
        temporal_candidates=(
            None
            if payload.get("temporal_candidates") is None
            else int(payload["temporal_candidates"])
        ),
    )


def load_expert_prototype_bank(path: str | Path) -> ExpertPrototypeBank:
    """Load a prototype bank without permitting arbitrary pickle globals."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _expert_prototype_from_payload(payload)


def _capture_state_samples(
    policy: Any,
    conflict: PrefixSnapshot,
    correct: PrefixSnapshot,
    noise: Tensor,
    spec: ExpertSteeringSpec,
) -> dict[int, tuple[Tensor, Tensor]]:
    model = policy.model
    layers, final_norm = expert_layers(policy)
    if spec.layer >= len(layers):
        raise ValueError("KNN samples require a valid expert layer")
    if spec.interface == "expert_output" and spec.layer != len(layers) - 1:
        raise ValueError("Expert-output samples require the final expert layer")
    states = expert_flow_states(model, conflict, noise)
    dt = -1.0 / int(model.config.num_steps)
    positions = list(spec.token_positions)
    result: dict[int, tuple[Tensor, Tensor]] = {}
    for flow_step in range(int(model.config.num_steps)):
        timestep = torch.full(
            (1,),
            1.0 + flow_step * dt,
            dtype=torch.float32,
            device=noise.device,
        )
        if spec.interface == "expert_output":
            _, conflict_activation = capture_expert_output(
                model, conflict, states[flow_step], timestep, final_norm
            )
            _, correct_activation = capture_expert_output(
                model, correct, states[flow_step], timestep, final_norm
            )
        else:
            _, conflict_residuals = capture_expert_post_residuals(
                model,
                conflict,
                states[flow_step],
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=(spec.layer,),
            )
            _, correct_residuals = capture_expert_post_residuals(
                model,
                correct,
                states[flow_step],
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=(spec.layer,),
            )
            conflict_activation = conflict_residuals[spec.layer]
            correct_activation = correct_residuals[spec.layer]
        feature = conflict_activation[:, positions].float().reshape(1, -1)
        target = (
            (correct_activation[:, positions] - conflict_activation[:, positions])
            .float()
            .reshape(1, -1)
        )
        result[flow_step] = (feature.cpu(), target.cpu())
    return result


def _knn_prediction(
    feature: Tensor,
    prototype_features: Tensor,
    prototype_directions: Tensor,
    neighbors: int,
    *,
    prototype_environment_steps: Tensor | None = None,
    query_environment_step: int | None = None,
    temporal_candidates: int | None = None,
) -> Tensor:
    if feature.ndim != 2 or feature.shape[0] != 1:
        raise ValueError("KNN query must have shape [1, hidden]")
    if prototype_features.shape != prototype_directions.shape:
        raise ValueError("KNN prototype feature and target shapes must match")
    if not 0 < neighbors <= prototype_features.shape[0]:
        raise ValueError("KNN neighbor count is outside the prototype bank")
    if temporal_candidates is not None:
        if prototype_environment_steps is None or query_environment_step is None:
            raise ValueError("Temporal KNN requires prototype and query environment steps")
        if prototype_environment_steps.ndim != 1 or (
            prototype_environment_steps.shape[0] != prototype_features.shape[0]
        ):
            raise ValueError("Prototype environment steps must match the prototype bank")
        if temporal_candidates < neighbors:
            raise ValueError("Temporal candidate count must be at least K")
        candidate_count = min(temporal_candidates, prototype_features.shape[0])
        temporal_distance = (
            prototype_environment_steps.to(torch.int64) - query_environment_step
        ).abs()
        temporal_indices = torch.topk(-temporal_distance.float(), k=candidate_count).indices
        prototype_features = prototype_features[temporal_indices]
        prototype_directions = prototype_directions[temporal_indices]
    query = torch.nn.functional.normalize(feature.float(), dim=1)
    keys = torch.nn.functional.normalize(prototype_features.float(), dim=1)
    similarities = query @ keys.transpose(0, 1)
    indices = torch.topk(similarities, k=neighbors, dim=1).indices[0]
    return prototype_directions[indices].mean(dim=0, keepdim=True)


def select_knn_hyperparameters(
    features: Mapping[int, Tensor],
    directions: Mapping[int, Tensor],
    episode_indices: Mapping[int, Tensor],
    candidates: tuple[int, ...],
    *,
    environment_steps: Mapping[int, Tensor] | None = None,
    temporal_candidate_counts: tuple[int | None, ...] = (None,),
) -> tuple[
    int,
    float,
    int | None,
    tuple[KNNCrossValidationRow, ...],
]:
    """Select K and one scale by leave-one-training-scene-out prediction."""

    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError("K candidates must be positive")
    if not temporal_candidate_counts or any(
        count is not None and count <= 0 for count in temporal_candidate_counts
    ):
        raise ValueError("Temporal candidate counts must be positive or None")
    if any(count is not None for count in temporal_candidate_counts) and (
        environment_steps is None
    ):
        raise ValueError("Temporal selection requires environment-step metadata")
    rows: list[KNNCrossValidationRow] = []
    for temporal_candidates in temporal_candidate_counts:
        for neighbors in candidates:
            if temporal_candidates is not None and temporal_candidates < neighbors:
                continue
            predictions: list[Tensor] = []
            targets: list[Tensor] = []
            for step in sorted(features):
                step_features = features[step]
                step_directions = directions[step]
                step_episodes = episode_indices[step]
                step_environment_steps = (
                    environment_steps[step] if environment_steps is not None else None
                )
                for index in range(step_features.shape[0]):
                    eligible = step_episodes != step_episodes[index]
                    eligible_count = int(eligible.sum())
                    if eligible_count < neighbors:
                        raise ValueError("K exceeds leave-one-scene-out training prototypes")
                    predictions.append(
                        _knn_prediction(
                            step_features[index : index + 1],
                            step_features[eligible],
                            step_directions[eligible],
                            neighbors,
                            prototype_environment_steps=(
                                step_environment_steps[eligible]
                                if step_environment_steps is not None
                                else None
                            ),
                            query_environment_step=(
                                int(step_environment_steps[index])
                                if step_environment_steps is not None
                                else None
                            ),
                            temporal_candidates=temporal_candidates,
                        )
                    )
                    targets.append(step_directions[index : index + 1])
            prediction = torch.cat(predictions).double()
            target = torch.cat(targets).double()
            denominator = torch.sum(prediction * prediction)
            if float(denominator) <= 1e-20:
                raise ValueError("KNN cross-validation predictions are degenerate")
            scale = float(torch.sum(prediction * target) / denominator)
            residual = scale * prediction - target
            target_energy = torch.sum(target * target)
            recovery = 1.0 - float(torch.sum(residual * residual) / target_energy)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    (scale * prediction).reshape(-1), target.reshape(-1), dim=0
                )
            )
            rows.append(
                {
                    "neighbors": neighbors,
                    "temporal_candidates": temporal_candidates,
                    "scale": scale,
                    "loo_target_mse_recovery": recovery,
                    "loo_target_cosine": cosine,
                }
            )
    best = max(
        rows,
        key=lambda row: (
            float(row["loo_target_mse_recovery"]),
            -int(row["neighbors"]),
            0 if row["temporal_candidates"] is None else -int(row["temporal_candidates"]),
        ),
    )
    return (
        int(best["neighbors"]),
        float(best["scale"]),
        None if best["temporal_candidates"] is None else int(best["temporal_candidates"]),
        tuple(rows),
    )


@dataclass(frozen=True)
class PrototypeTrainingEpisode:
    episode_index: int
    rollout_noise_seed: int
    success: bool
    steps: int
    sampled_environment_steps: tuple[int, ...]


@dataclass(frozen=True)
class ExpertPrototypeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    training_episode_indices: tuple[int, ...]
    spec: dict[str, object]
    sample_stride: int
    max_steps: int
    noise_seed: int
    rollout_noise_seeds: tuple[int, ...]
    prototype_noise_seeds: tuple[int, ...]
    simulator_seed: int
    scene_condition: SceneCondition | None
    hidden_object: str | None
    native_prompt_role: str
    samples_per_flow_step: int
    selected_neighbors: int
    selected_scale: float
    selected_temporal_candidates: int | None
    cross_validation: tuple[KNNCrossValidationRow, ...]
    episodes: tuple[PrototypeTrainingEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["cross_validation"] = list(self.cross_validation)
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def fit_expert_prototype_bank(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    training_episode_indices: tuple[int, ...],
    expert_layer: int = 15,
    expert_interface: Literal["post_residual", "expert_output"] = "post_residual",
    action_token_positions: tuple[int, ...] = (0,),
    sample_stride: int = 20,
    neighbor_candidates: tuple[int, ...] = (1, 3, 5),
    temporal_candidate_counts: tuple[int | None, ...] = (None,),
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    rollout_noise_seeds: tuple[int, ...] | None = None,
    prototype_noise_seeds: tuple[int, ...] | None = None,
    simulator_seed: int = 42,
    max_steps: int = 220,
    scene_condition: SceneCondition | None = None,
    hidden_object: str | None = None,
    progress: Callable[[int, int, bool], None] | None = None,
) -> tuple[ExpertPrototypeBank, ExpertPrototypeFitReport]:
    """Collect successful training trajectories and fit a LOO-selected KNN bank."""

    if not training_episode_indices or len(set(training_episode_indices)) != len(
        training_episode_indices
    ):
        raise ValueError("Training episodes must be nonempty and unique")
    if sample_stride <= 0 or max_steps <= 0:
        raise ValueError("Sample stride and maximum steps must be positive")
    if set(training_episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if scene_condition is not None and hidden_object is not None:
        raise ValueError("Spatial and hidden-object interventions are mutually exclusive")
    if scene_condition is not None:
        spatial_support_task(task_id)
    if hidden_object is not None:
        task = object_conflict_task(task_id)
        if suite != "libero_object" or task.hidden_object != hidden_object:
            raise ValueError("Hidden object does not match the LIBERO Object task spec")
    active_prototype_noise_seeds = (
        (noise_seed,) if prototype_noise_seeds is None else prototype_noise_seeds
    )
    active_rollout_noise_seeds = (
        (noise_seed,) if rollout_noise_seeds is None else rollout_noise_seeds
    )
    if not active_prototype_noise_seeds or len(set(active_prototype_noise_seeds)) != len(
        active_prototype_noise_seeds
    ):
        raise ValueError("Prototype noise seeds must be nonempty and unique")
    if not active_rollout_noise_seeds or len(set(active_rollout_noise_seeds)) != len(
        active_rollout_noise_seeds
    ):
        raise ValueError("Rollout noise seeds must be nonempty and unique")
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    torch_device = torch.device(device)
    spec = ExpertSteeringSpec(
        f"l{expert_layer}_{expert_interface}_executed_action_all_steps",
        expert_layer,
        action_token_positions,
        expert_interface,
    )
    feature_lists: dict[int, list[Tensor]] = {
        step: [] for step in range(int(policy.model.config.num_steps))
    }
    direction_lists: dict[int, list[Tensor]] = {
        step: [] for step in range(int(policy.model.config.num_steps))
    }
    episode_lists: dict[int, list[int]] = {
        step: [] for step in range(int(policy.model.config.num_steps))
    }
    environment_step_lists: dict[int, list[int]] = {
        step: [] for step in range(int(policy.model.config.num_steps))
    }
    outcomes: list[PrototypeTrainingEpisode] = []
    for rollout_noise_seed in active_rollout_noise_seeds:
        for episode_index in training_episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            expected_native_prompt = (
                conflict_prompt if scene_condition == "conflict" else correct_prompt
            )
            if native_prompt != expected_native_prompt:
                environment.close()
                role = "conflict" if scene_condition == "conflict" else "correct"
                raise ValueError(f"The {role} prompt must exactly match the native instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            if scene_condition is not None:
                observation, _ = apply_unique_bowl_scene(environment, condition=scene_condition)
            elif hidden_object is not None:
                observation, _ = apply_hidden_object_scene(
                    environment, object_name=hidden_object
                )
            step = 0
            success = False
            sampled_steps: list[int] = []
            try:
                while step < max_steps and not success:
                    correct = prefix_snapshot(
                        policy, prepare_libero_batch(adapter, observation, correct_prompt)
                    )
                    noise = fixed_noise(
                        policy,
                        rollout_noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    if step % sample_stride == 0:
                        conflict = prefix_snapshot(
                            policy,
                            prepare_libero_batch(adapter, observation, conflict_prompt),
                        )
                        for prototype_noise_seed in active_prototype_noise_seeds:
                            prototype_noise = fixed_noise(
                                policy,
                                prototype_noise_seed
                                + rollout_noise_seed * 1_000_000
                                + episode_index * 10_000
                                + step,
                                torch_device,
                            )
                            samples = _capture_state_samples(
                                policy, conflict, correct, prototype_noise, spec
                            )
                            for flow_step, (feature, direction) in samples.items():
                                feature_lists[flow_step].append(feature)
                                direction_lists[flow_step].append(direction)
                                episode_lists[flow_step].append(episode_index)
                                environment_step_lists[flow_step].append(step)
                        sampled_steps.append(step)
                        if progress is not None:
                            progress(episode_index, step, True)
                    chunk = _sample_natural(policy, correct, noise)
                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    if progress is not None and step % sample_stride == 0:
                        progress(episode_index, step, False)
                    if terminated or truncated:
                        break
            finally:
                environment.close()
            outcome = PrototypeTrainingEpisode(
                episode_index=episode_index,
                rollout_noise_seed=rollout_noise_seed,
                success=success,
                steps=step,
                sampled_environment_steps=tuple(sampled_steps),
            )
            outcomes.append(outcome)
            if not success:
                raise RuntimeError(
                    f"Correct policy did not solve training episode {episode_index} "
                    f"at rollout noise {rollout_noise_seed}; prototype bank requires "
                    "successful trajectories"
                )
    features = {step: torch.cat(values) for step, values in feature_lists.items()}
    directions = {step: torch.cat(values) for step, values in direction_lists.items()}
    episodes = {
        step: torch.tensor(values, dtype=torch.int64) for step, values in episode_lists.items()
    }
    environment_steps = {
        step: torch.tensor(values, dtype=torch.int64)
        for step, values in environment_step_lists.items()
    }
    neighbors, scale, temporal_candidates, cross_validation = select_knn_hyperparameters(
        features,
        directions,
        episodes,
        neighbor_candidates,
        environment_steps=environment_steps,
        temporal_candidate_counts=temporal_candidate_counts,
    )
    bank = ExpertPrototypeBank(
        spec=spec,
        num_steps=int(policy.model.config.num_steps),
        features=features,
        directions=directions,
        episode_indices=episodes,
        environment_steps=environment_steps,
        neighbors=neighbors,
        scale=scale,
        temporal_candidates=temporal_candidates,
    )
    report = ExpertPrototypeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        training_episode_indices=training_episode_indices,
        spec=asdict(spec),
        sample_stride=sample_stride,
        max_steps=max_steps,
        noise_seed=noise_seed,
        rollout_noise_seeds=active_rollout_noise_seeds,
        prototype_noise_seeds=active_prototype_noise_seeds,
        simulator_seed=simulator_seed,
        scene_condition=scene_condition,
        hidden_object=hidden_object,
        native_prompt_role=("conflict" if scene_condition == "conflict" else "correct"),
        samples_per_flow_step=int(features[0].shape[0]),
        selected_neighbors=neighbors,
        selected_scale=scale,
        selected_temporal_candidates=temporal_candidates,
        cross_validation=cross_validation,
        episodes=tuple(outcomes),
    )
    return bank, report


def merge_expert_prototype_banks(
    banks: tuple[ExpertPrototypeBank, ...],
    *,
    neighbor_candidates: tuple[int, ...] = (1, 3, 5),
    temporal_candidate_counts: tuple[int | None, ...] = (None,),
) -> tuple[ExpertPrototypeBank, tuple[KNNCrossValidationRow, ...]]:
    """Pool compatible training banks and reselect hyperparameters by scene group."""

    if len(banks) < 2:
        raise ValueError("Merging requires at least two prototype banks")
    reference = banks[0]
    if any(
        bank.spec != reference.spec or bank.num_steps != reference.num_steps for bank in banks[1:]
    ):
        raise ValueError("Prototype banks must share one expert locus and flow schedule")
    features = {
        step: torch.cat([bank.features[step] for bank in banks])
        for step in range(reference.num_steps)
    }
    directions = {
        step: torch.cat([bank.directions[step] for bank in banks])
        for step in range(reference.num_steps)
    }
    episodes = {
        step: torch.cat([bank.episode_indices[step] for bank in banks])
        for step in range(reference.num_steps)
    }
    environment_steps = {
        step: torch.cat([bank.environment_steps[step] for bank in banks])
        for step in range(reference.num_steps)
    }
    neighbors, scale, temporal_candidates, rows = select_knn_hyperparameters(
        features,
        directions,
        episodes,
        neighbor_candidates,
        environment_steps=environment_steps,
        temporal_candidate_counts=temporal_candidate_counts,
    )
    return (
        ExpertPrototypeBank(
            spec=reference.spec,
            num_steps=reference.num_steps,
            features=features,
            directions=directions,
            episode_indices=episodes,
            environment_steps=environment_steps,
            neighbors=neighbors,
            scale=scale,
            temporal_candidates=temporal_candidates,
        ),
        rows,
    )


def _sample_natural(policy: Any, snapshot: PrefixSnapshot, noise: Tensor) -> Tensor:
    model = policy.model
    x_t = noise.clone()
    dt = -1.0 / int(model.config.num_steps)
    with torch.inference_mode():
        for flow_step in range(int(model.config.num_steps)):
            timestep = torch.full(
                (x_t.shape[0],),
                1.0 + flow_step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks,
                past_key_values=snapshot.cache,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


class AdaptiveExpertKNN:
    """Predict and add a residual correction from current conflict activation."""

    def __init__(
        self,
        bank: ExpertPrototypeBank,
        flow_step: int,
        *,
        environment_step: int | None = None,
        random_seed: int | None = None,
    ):
        self.bank = bank
        self.flow_step = flow_step
        self.environment_step = environment_step
        self.random_seed = random_seed
        self.residual_input: Tensor | None = None
        self.calls = 0

    def capture_residual(self, _module: Any, inputs: tuple[object, ...]) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor residual input")
        if self.residual_input is not None:
            raise RuntimeError("Expert residual was captured more than once")
        self.residual_input = inputs[0].detach().clone()

    def add_prediction(self, _module: Any, _inputs: tuple[object, ...], output: object) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("Expected a tensor expert MLP output")
        if self.bank.spec.interface == "post_residual" and self.residual_input is None:
            raise RuntimeError("Expert MLP ran before residual capture")
        if self.calls:
            raise RuntimeError("Expert KNN hook was called more than once")
        self.calls += 1
        activation = (
            output
            if self.bank.spec.interface == "expert_output"
            else cast(Tensor, self.residual_input) + output
        )
        positions = list(self.bank.spec.token_positions)
        selected = activation[:, positions]
        query = selected.float().reshape(1, -1)
        prediction = _knn_prediction(
            query,
            self.bank.features[self.flow_step].to(query.device),
            self.bank.directions[self.flow_step].to(query.device),
            self.bank.neighbors,
            prototype_environment_steps=self.bank.environment_steps[self.flow_step],
            query_environment_step=self.environment_step,
            temporal_candidates=self.bank.temporal_candidates,
        ).reshape_as(selected)
        if self.random_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            random_direction = torch.randn(
                prediction.shape,
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            )
            prediction_norm = torch.linalg.vector_norm(prediction.float().cpu())
            random_direction = random_direction * (
                prediction_norm / torch.linalg.vector_norm(random_direction).clamp_min(1e-12)
            )
            prediction = random_direction.to(prediction)
        updated = output.clone()
        updated[:, positions] = output[:, positions] + prediction.to(output.dtype) * self.bank.scale
        return updated

    def assert_applied(self) -> None:
        residual_missing = (
            self.bank.spec.interface == "post_residual" and self.residual_input is None
        )
        if residual_missing or self.calls != 1:
            raise RuntimeError("Adaptive expert KNN was not applied exactly once")


def sample_with_expert_knn(
    policy: Any,
    snapshot: PrefixSnapshot,
    noise: Tensor,
    bank: ExpertPrototypeBank,
    *,
    environment_step: int | None = None,
    direction_sign: float = 1.0,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate an action chunk using online KNN directions from a frozen bank."""

    if direction_sign not in {-1.0, 1.0}:
        raise ValueError("Direction sign must be +1 or -1")
    model = policy.model
    layers, final_norm = expert_layers(policy)
    if bank.spec.layer >= len(layers) or bank.num_steps != int(model.config.num_steps):
        raise ValueError("Prototype bank is incompatible with the policy")
    if bank.spec.interface == "expert_output" and bank.spec.layer != len(layers) - 1:
        raise ValueError("Expert-output steering requires the final expert layer")
    active_bank = bank
    if direction_sign < 0:
        active_bank = ExpertPrototypeBank(
            spec=bank.spec,
            num_steps=bank.num_steps,
            features=bank.features,
            directions={step: -value for step, value in bank.directions.items()},
            episode_indices=bank.episode_indices,
            environment_steps=bank.environment_steps,
            neighbors=bank.neighbors,
            scale=bank.scale,
            temporal_candidates=bank.temporal_candidates,
        )
    x_t = noise.clone()
    dt = -1.0 / bank.num_steps
    for flow_step in range(bank.num_steps):
        timestep = torch.full(
            (x_t.shape[0],),
            1.0 + flow_step * dt,
            dtype=torch.float32,
            device=x_t.device,
        )
        intervention = AdaptiveExpertKNN(
            active_bank,
            flow_step,
            environment_step=environment_step,
            random_seed=(None if random_seed is None else random_seed + flow_step),
        )
        if bank.spec.interface == "expert_output":
            residual_handle = None
            prediction_handle = final_norm.register_forward_hook(intervention.add_prediction)
        else:
            layer = layers[bank.spec.layer]
            residual_handle = layer.post_attention_layernorm.register_forward_pre_hook(
                intervention.capture_residual
            )
            prediction_handle = layer.mlp.register_forward_hook(intervention.add_prediction)
        try:
            with torch.inference_mode():
                velocity = model.denoise_step(
                    prefix_pad_masks=snapshot.pad_masks,
                    past_key_values=snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                )
        finally:
            if residual_handle is not None:
                residual_handle.remove()
            prediction_handle.remove()
        intervention.assert_applied()
        x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


ExpertKNNCondition = Literal[
    "correct",
    "conflict",
    "steered",
    "wrong_sign",
    "random",
    "aligned_steered",
]
VALID_EXPERT_KNN_CONDITIONS = frozenset(
    {"correct", "conflict", "steered", "wrong_sign", "random", "aligned_steered"}
)


def validate_expert_knn_conditions(
    conditions: tuple[str, ...],
) -> tuple[ExpertKNNCondition, ...]:
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Expert KNN conditions must be nonempty and unique")
    unknown = set(conditions) - VALID_EXPERT_KNN_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown expert KNN conditions: {sorted(unknown)}")
    return cast(tuple[ExpertKNNCondition, ...], conditions)


@dataclass(frozen=True)
class ExpertKNNEpisode:
    episode_index: int
    condition: ExpertKNNCondition
    success: bool
    steps: int
    steering_calls: int
    scene_condition: SceneCondition | None
    prompt: str
    native_prompt: str


@dataclass(frozen=True)
class ExpertKNNEvaluationReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    suite: str
    task_id: int
    conflict_prompt: str
    episode_indices: tuple[int, ...]
    conditions: tuple[ExpertKNNCondition, ...]
    spec: dict[str, object]
    neighbors: int
    scale: float
    noise_seed: int
    simulator_seed: int
    max_steps: int
    scene_condition: SceneCondition | None
    hidden_object: str | None
    random_seed: int
    evaluation_target_activations_used: int
    steered_runtime_correct_prompt_forwards: int
    runtime_prompt_rewrites: int
    language_token_interventions: int
    episodes: tuple[ExpertKNNEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def run_expert_knn_closed_loop(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    conflict_prompt: str,
    episode_indices: tuple[int, ...],
    conditions: tuple[ExpertKNNCondition, ...] = ("steered",),
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    simulator_seed: int = 42,
    max_steps: int = 220,
    random_seed: int = 30_007,
    scene_condition: SceneCondition | None = None,
    hidden_object: str | None = None,
    progress: Callable[[ExpertKNNEpisode], None] | None = None,
) -> ExpertKNNEvaluationReport:
    """Evaluate a frozen trajectory-conditioned bank without target activations."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be nonempty and unique")
    if max_steps <= 0:
        raise ValueError("Maximum rollout steps must be positive")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if scene_condition is not None and hidden_object is not None:
        raise ValueError("Spatial and hidden-object interventions are mutually exclusive")
    if scene_condition is not None:
        spatial_support_task(task_id)
    if hidden_object is not None:
        task = object_conflict_task(task_id)
        if suite != "libero_object" or task.hidden_object != hidden_object:
            raise ValueError("Hidden object does not match the LIBERO Object task spec")
        if conflict_prompt.casefold() != task.conflict_prompt.casefold():
            raise ValueError("Conflict prompt does not match the object conflict task spec")
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    bank = load_expert_prototype_bank(bank_path)
    torch_device = torch.device(device)
    outcomes: list[ExpertKNNEpisode] = []
    for episode_index in episode_indices:
        for condition in conditions:
            environment, native_prompt, _ = adapter.create_env(
                task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            if scene_condition is not None and native_prompt != conflict_prompt:
                environment.close()
                raise ValueError(
                    "Conflict prompt must exactly match the native instruction when "
                    "using visual scene interventions"
                )
            if hidden_object is not None and (
                native_prompt.casefold()
                != object_conflict_task(task_id).native_prompt.casefold()
            ):
                environment.close()
                raise ValueError("Native prompt does not match the object conflict task spec")
            step = 0
            success = False
            steering_calls = 0
            try:
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                if scene_condition is not None:
                    observation, _ = apply_unique_bowl_scene(environment, condition=scene_condition)
                elif hidden_object is not None:
                    observation, _ = apply_hidden_object_scene(
                        environment, object_name=hidden_object
                    )
                while step < max_steps and not success:
                    prompt = (
                        native_prompt
                        if condition in {"correct", "aligned_steered"}
                        else conflict_prompt
                    )
                    snapshot = prefix_snapshot(
                        policy, prepare_libero_batch(adapter, observation, prompt)
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    if condition in {"correct", "conflict"}:
                        chunk = _sample_natural(policy, snapshot, noise)
                    else:
                        chunk = sample_with_expert_knn(
                            policy,
                            snapshot,
                            noise,
                            bank,
                            environment_step=step,
                            direction_sign=-1.0 if condition == "wrong_sign" else 1.0,
                            random_seed=(
                                random_seed + episode_index * 10_000 + step * 10
                                if condition == "random"
                                else None
                            ),
                        )
                        steering_calls += 1
                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    if terminated or truncated:
                        break
            finally:
                environment.close()
            outcome = ExpertKNNEpisode(
                episode_index=episode_index,
                condition=condition,
                success=success,
                steps=step,
                steering_calls=steering_calls,
                scene_condition=scene_condition,
                prompt=prompt,
                native_prompt=native_prompt,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(outcome)
    return ExpertKNNEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        suite=suite,
        task_id=task_id,
        conflict_prompt=conflict_prompt,
        episode_indices=episode_indices,
        conditions=conditions,
        spec=asdict(bank.spec),
        neighbors=bank.neighbors,
        scale=bank.scale,
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        max_steps=max_steps,
        scene_condition=scene_condition,
        hidden_object=hidden_object,
        random_seed=random_seed,
        evaluation_target_activations_used=0,
        steered_runtime_correct_prompt_forwards=0,
        runtime_prompt_rewrites=0,
        language_token_interventions=0,
        episodes=tuple(outcomes),
    )
