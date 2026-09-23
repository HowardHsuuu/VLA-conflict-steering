"""Source-free steering at a causally localized SmolVLA action-expert locus."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from causal_vla.causal_trace import TokenPatch, recovery_score
from causal_vla.causal_trace_runtime import PrefixSnapshot, prefix_snapshot
from causal_vla.expert_trace_runtime import (
    capture_expert_post_residuals,
    expert_flow_states,
    expert_layers,
    patched_expert_velocity,
)
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.interventions import ActivationCapture, ActivationReplace, ResidualStreamAdd
from causal_vla.residual_runtime import decode_action
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch


@dataclass(frozen=True)
class ExpertSteeringSpec:
    """One action-expert layer and token set selected by causal tracing."""

    name: str
    layer: int
    token_positions: tuple[int, ...]
    interface: Literal["post_residual", "expert_output"] = "post_residual"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Expert-steering spec name must not be empty")
        if self.layer < 0:
            raise ValueError("Expert-steering layer must be nonnegative")
        if not self.token_positions or len(set(self.token_positions)) != len(self.token_positions):
            raise ValueError("Action-token positions must be nonempty and unique")
        if any(position < 0 for position in self.token_positions):
            raise ValueError("Action-token positions must be nonnegative")
        if self.interface not in {"post_residual", "expert_output"}:
            raise ValueError("Unknown expert-steering interface")


StepDirectionMap = dict[int, Tensor]


class ExpertOutputAdd:
    """Add one cached direction to the normalized final expert output."""

    def __init__(self, direction: Tensor, positions: tuple[int, ...]):
        self.direction = direction.detach()
        self.positions = positions
        self.calls = 0

    def __call__(
        self, _module: Any, _inputs: tuple[object, ...], output: object
    ) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("Expected a tensor expert output")
        if self.calls:
            raise RuntimeError("Expert output normalization was called more than once")
        self.calls += 1
        positions = list(self.positions)
        selected = output[:, positions]
        direction = self.direction.to(device=output.device, dtype=output.dtype)
        if direction.shape != selected.shape:
            raise ValueError("Expert output direction shape does not match selected tokens")
        updated = output.clone()
        updated[:, positions] = selected + direction
        return updated

    def assert_applied(self) -> None:
        """Require one residual addition."""

        if self.calls != 1:
            raise RuntimeError("Expert output direction was not applied exactly once")


def capture_expert_output(
    model: Any,
    snapshot: PrefixSnapshot,
    x_t: Tensor,
    timestep: Tensor,
    final_norm: Any,
) -> tuple[Tensor, Tensor]:
    """Capture the normalized expert representation consumed by action projection."""

    capture = ActivationCapture()
    handle = final_norm.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks,
                past_key_values=snapshot.cache,
                x_t=x_t,
                timestep=timestep,
            ).detach()
    finally:
        handle.remove()
    if len(capture.records) != 1:
        raise RuntimeError("Expected one normalized expert output per denoising call")
    return velocity, capture.records[0]


def mean_expert_directions(
    scene_directions: Mapping[int, StepDirectionMap],
) -> StepDirectionMap:
    """Average each flow-time direction over training scenes only."""

    if not scene_directions:
        raise ValueError("At least one training scene direction is required")
    expected_steps = set(next(iter(scene_directions.values())))
    if not expected_steps:
        raise ValueError("Training directions must contain at least one flow step")
    if any(set(directions) != expected_steps for directions in scene_directions.values()):
        raise ValueError("All training scenes must contain the same flow steps")
    return {
        step: torch.stack([directions[step] for directions in scene_directions.values()]).mean(
            dim=0
        )
        for step in sorted(expected_steps)
    }


def _scaled_directions(directions: Mapping[int, Tensor], scale: float) -> StepDirectionMap:
    if not math.isfinite(scale):
        raise ValueError("Expert-steering scale must be finite")
    return {step: direction * scale for step, direction in directions.items()}


def capture_scene_expert_directions(
    policy: Any,
    conflict: PrefixSnapshot,
    correct: PrefixSnapshot,
    noise: Tensor,
    spec: ExpertSteeringSpec,
) -> StepDirectionMap:
    """Capture correct-minus-conflict expert outputs at matched conflict flow states."""

    model = policy.model
    layers, final_norm = expert_layers(policy)
    if spec.layer >= len(layers):
        raise ValueError("Expert-steering layer is outside the decoder")
    if spec.interface == "expert_output" and spec.layer != len(layers) - 1:
        raise ValueError("Expert-output steering requires the final expert layer")
    if any(position >= int(model.config.chunk_size) for position in spec.token_positions):
        raise ValueError("Action-token position is outside the action chunk")
    states = expert_flow_states(model, conflict, noise)
    dt = -1.0 / int(model.config.num_steps)
    positions = list(spec.token_positions)
    directions: StepDirectionMap = {}
    for step in range(int(model.config.num_steps)):
        timestep = torch.full(
            (1,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=noise.device,
        )
        if spec.interface == "expert_output":
            _, conflict_output = capture_expert_output(
                model, conflict, states[step], timestep, final_norm
            )
            _, correct_output = capture_expert_output(
                model, correct, states[step], timestep, final_norm
            )
            directions[step] = (
                correct_output[:, positions] - conflict_output[:, positions]
            ).detach()
        else:
            _, conflict_residuals = capture_expert_post_residuals(
                model,
                conflict,
                states[step],
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=(spec.layer,),
            )
            _, correct_residuals = capture_expert_post_residuals(
                model,
                correct,
                states[step],
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=(spec.layer,),
            )
            directions[step] = (
                correct_residuals[spec.layer][:, positions]
                - conflict_residuals[spec.layer][:, positions]
            ).detach()
    return directions


def sample_with_expert_directions(
    policy: Any,
    snapshot: PrefixSnapshot,
    noise: Tensor,
    spec: ExpertSteeringSpec | None,
    directions: Mapping[int, Tensor] | None,
) -> Tensor:
    """Integrate an action chunk with one learned expert-output direction per flow step."""

    if (spec is None) != (directions is None):
        raise ValueError("Expert-steering spec and directions must be provided together")
    model = policy.model
    layers, final_norm = expert_layers(policy)
    if spec is not None and spec.layer >= len(layers):
        raise ValueError("Expert-steering layer is outside the decoder")
    if (
        spec is not None
        and spec.interface == "expert_output"
        and spec.layer != len(layers) - 1
    ):
        raise ValueError("Expert-output steering requires the final expert layer")
    expected_steps = set(range(int(model.config.num_steps)))
    if directions is not None and set(directions) != expected_steps:
        raise ValueError("Expert directions must cover every flow step exactly once")
    x_t = noise.clone()
    dt = -1.0 / int(model.config.num_steps)
    with torch.inference_mode():
        for step in range(int(model.config.num_steps)):
            timestep = torch.full(
                (x_t.shape[0],),
                1.0 + step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            intervention: ExpertOutputAdd | ResidualStreamAdd | None = None
            handle = None
            if spec is not None and directions is not None:
                if spec.interface == "expert_output":
                    intervention = ExpertOutputAdd(directions[step], spec.token_positions)
                    handle = final_norm.register_forward_hook(intervention)
                else:
                    intervention = ResidualStreamAdd(
                        (directions[step],), positions=spec.token_positions
                    )
                    handle = layers[spec.layer].mlp.register_forward_hook(intervention)
            try:
                velocity = model.denoise_step(
                    prefix_pad_masks=snapshot.pad_masks,
                    past_key_values=snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                )
            finally:
                if handle is not None:
                    handle.remove()
            if intervention is not None:
                if isinstance(intervention, ExpertOutputAdd):
                    intervention.assert_applied()
                else:
                    intervention.assert_consumed()
            x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


def sample_with_oracle_expert_patch(
    policy: Any,
    conflict: PrefixSnapshot,
    correct: PrefixSnapshot,
    noise: Tensor,
    spec: ExpertSteeringSpec,
) -> Tensor:
    """Integrate with exact current-state expert outputs for upper-bound diagnosis."""

    model = policy.model
    layers, final_norm = expert_layers(policy)
    if spec.layer >= len(layers):
        raise ValueError("Expert-steering layer is outside the decoder")
    if spec.interface == "expert_output" and spec.layer != len(layers) - 1:
        raise ValueError("Expert-output steering requires the final expert layer")
    x_t = noise.clone()
    dt = -1.0 / int(model.config.num_steps)
    for step in range(int(model.config.num_steps)):
        timestep = torch.full(
            (x_t.shape[0],),
            1.0 + step * dt,
            dtype=torch.float32,
            device=x_t.device,
        )
        if spec.interface == "expert_output":
            _, correct_output = capture_expert_output(
                model, correct, x_t, timestep, final_norm
            )
            replacement = ActivationReplace((correct_output,), positions=spec.token_positions)
            handle = final_norm.register_forward_hook(replacement)
            try:
                with torch.inference_mode():
                    velocity = model.denoise_step(
                        prefix_pad_masks=conflict.pad_masks,
                        past_key_values=conflict.cache,
                        x_t=x_t,
                        timestep=timestep,
                    )
            finally:
                handle.remove()
            replacement.assert_consumed()
        else:
            _, correct_residuals = capture_expert_post_residuals(
                model,
                correct,
                x_t,
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=(spec.layer,),
            )
            velocity = patched_expert_velocity(
                model,
                conflict,
                x_t,
                timestep,
                layers=layers,
                expert_layer=spec.layer,
                source=correct_residuals[spec.layer],
                patches=(TokenPatch("oracle", spec.token_positions),),
            )
        x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


def expert_direction_diagnostics(
    scene_directions: Mapping[int, StepDirectionMap],
) -> tuple[dict[str, object], ...]:
    """Measure whether the localized residual delta is stable across scenes."""

    if len(scene_directions) < 2:
        raise ValueError("Direction diagnostics require at least two scenes")
    steps = sorted(next(iter(scene_directions.values())))
    rows: list[dict[str, object]] = []
    for step in steps:
        vectors = [
            directions[step].float().reshape(-1).cpu() for directions in scene_directions.values()
        ]
        cosines: list[float] = []
        for left_index, left in enumerate(vectors):
            for right in vectors[left_index + 1 :]:
                cosines.append(float(torch.nn.functional.cosine_similarity(left, right, dim=0)))
        rows.append(
            {
                "denoising_step": step,
                "pairwise_cosine_min": min(cosines),
                "pairwise_cosine_mean": sum(cosines) / len(cosines),
                "scene_delta_norms": [float(vector.norm()) for vector in vectors],
            }
        )
    return tuple(rows)


def fit_expert_direction_scale(
    policy: Any,
    training_pairs: Mapping[int, tuple[PrefixSnapshot, PrefixSnapshot]],
    spec: ExpertSteeringSpec,
    directions: Mapping[int, Tensor],
    *,
    noise_seed: int,
    device: torch.device,
) -> float:
    """Fit one global scale from training-scene integrated first actions."""

    if not training_pairs:
        raise ValueError("Scale fitting requires at least one training scene")
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for episode_index, (conflict, correct) in training_pairs.items():
        noise = fixed_noise(policy, noise_seed + episode_index, device)
        conflict_chunk = sample_with_expert_directions(policy, conflict, noise, None, None)
        correct_chunk = sample_with_expert_directions(policy, correct, noise, None, None)
        steered_chunk = sample_with_expert_directions(policy, conflict, noise, spec, directions)
        target = (correct_chunk[:, :1] - conflict_chunk[:, :1]).cpu().double()
        effect = (steered_chunk[:, :1] - conflict_chunk[:, :1]).cpu().double()
        numerator += torch.sum(effect * target)
        denominator += torch.sum(effect * effect)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-20:
        raise ValueError("Training steering effects are too small to fit a scale")
    scale = float(numerator / denominator)
    if not math.isfinite(scale):
        raise ValueError("Fitted expert-steering scale is not finite")
    return scale


@dataclass(frozen=True)
class ExpertSteeringScreenReport:
    """Disjoint-scene integrated-action evaluation of learned expert steering."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    training_episode_indices: tuple[int, ...]
    episode_indices: tuple[int, ...]
    spec: dict[str, object]
    noise_seed: int
    simulator_seed: int
    scene_condition: SceneCondition | None
    fitted_scale: float
    direction_diagnostics: tuple[dict[str, object], ...]
    rows: tuple[dict[str, object], ...]
    heldout_target_activations_used: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""

        payload = cast(dict[str, object], asdict(self))
        payload["direction_diagnostics"] = list(self.direction_diagnostics)
        payload["rows"] = list(self.rows)
        return payload


def _initial_snapshots(
    adapter: Any,
    policy: Any,
    episode_indices: tuple[int, ...],
    *,
    correct_prompt: str,
    conflict_prompt: str,
    suite: str,
    task_id: int,
    resolution: int,
    simulator_seed: int,
    scene_condition: SceneCondition | None = None,
) -> dict[int, tuple[PrefixSnapshot, PrefixSnapshot]]:
    pairs: dict[int, tuple[PrefixSnapshot, PrefixSnapshot]] = {}
    for episode_index in episode_indices:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            expected_native_prompt = (
                conflict_prompt if scene_condition == "conflict" else correct_prompt
            )
            if native_prompt != expected_native_prompt:
                role = "conflict" if scene_condition == "conflict" else "correct"
                raise ValueError(f"The {role} prompt must exactly match the native instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            if scene_condition is not None:
                observation, _ = apply_unique_bowl_scene(
                    environment, condition=scene_condition
                )
            conflict = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, conflict_prompt)
            )
            correct = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, correct_prompt)
            )
            pairs[episode_index] = (conflict, correct)
        finally:
            environment.close()
    return pairs


def run_expert_steering_screen(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    training_episode_indices: tuple[int, ...],
    episode_indices: tuple[int, ...],
    expert_layer: int = 15,
    action_token_positions: tuple[int, ...] = (0,),
    expert_interface: Literal["post_residual", "expert_output"] = "post_residual",
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 107,
    simulator_seed: int = 42,
    scene_condition: SceneCondition | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> ExpertSteeringScreenReport:
    """Learn step-specific directions on training scenes and screen disjoint scenes."""

    if not training_episode_indices or not episode_indices:
        raise ValueError("Training and evaluation episodes must be nonempty")
    if len(set(training_episode_indices)) != len(training_episode_indices):
        raise ValueError("Training episodes must be unique")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be unique")
    if set(training_episode_indices) & set(episode_indices):
        raise ValueError("Training and evaluation episodes must be disjoint")
    if set((*training_episode_indices, *episode_indices)) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if scene_condition is not None:
        spatial_support_task(task_id)
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
    pairs = _initial_snapshots(
        adapter,
        policy,
        (*training_episode_indices, *episode_indices),
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        suite=suite,
        task_id=task_id,
        resolution=resolution,
        simulator_seed=simulator_seed,
        scene_condition=scene_condition,
    )
    scene_directions: dict[int, StepDirectionMap] = {}
    for episode_index in training_episode_indices:
        conflict, correct = pairs[episode_index]
        scene_directions[episode_index] = capture_scene_expert_directions(
            policy,
            conflict,
            correct,
            fixed_noise(policy, noise_seed + episode_index, torch_device),
            spec,
        )
        if progress is not None:
            progress(episode_index, "captured_training_direction")
    directions = mean_expert_directions(scene_directions)
    training_pairs = {episode: pairs[episode] for episode in training_episode_indices}
    fitted_scale = fit_expert_direction_scale(
        policy,
        training_pairs,
        spec,
        directions,
        noise_seed=noise_seed,
        device=torch_device,
    )
    calibrated = _scaled_directions(directions, fitted_scale)
    rows: list[dict[str, object]] = []
    action_dim = int(policy.config.action_feature.shape[0])
    for episode_index in episode_indices:
        conflict, correct = pairs[episode_index]
        noise = fixed_noise(policy, noise_seed + episode_index, torch_device)
        conflict_chunk = sample_with_expert_directions(policy, conflict, noise, None, None)
        correct_chunk = sample_with_expert_directions(policy, correct, noise, None, None)
        for condition, active_directions in (
            ("unscaled", directions),
            ("calibrated", calibrated),
        ):
            steered_chunk = sample_with_expert_directions(
                policy, conflict, noise, spec, active_directions
            )
            executed = recovery_score(
                conflict_chunk[:, :1, :action_dim],
                correct_chunk[:, :1, :action_dim],
                steered_chunk[:, :1, :action_dim],
            )
            chunk = recovery_score(conflict_chunk, correct_chunk, steered_chunk)
            rows.append(
                {
                    "episode_index": episode_index,
                    "condition": condition,
                    "executed_directional_recovery": executed.directional_recovery,
                    "executed_mse_recovery": executed.mse_recovery,
                    "executed_effect_l2": executed.effect_l2,
                    "executed_conflict_correct_l2": executed.conflict_correct_l2,
                    "chunk_directional_recovery": chunk.directional_recovery,
                    "chunk_mse_recovery": chunk.mse_recovery,
                    "chunk_effect_l2": chunk.effect_l2,
                    "chunk_conflict_correct_l2": chunk.conflict_correct_l2,
                }
            )
        if progress is not None:
            progress(episode_index, "screened_evaluation_scene")
    return ExpertSteeringScreenReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        training_episode_indices=training_episode_indices,
        episode_indices=episode_indices,
        spec=asdict(spec),
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        scene_condition=scene_condition,
        fitted_scale=fitted_scale,
        direction_diagnostics=expert_direction_diagnostics(scene_directions),
        rows=tuple(rows),
        heldout_target_activations_used=0,
    )


ExpertSteeringCondition = Literal[
    "correct", "conflict", "steered", "wrong_sign", "oracle_steered"
]
VALID_EXPERT_STEERING_CONDITIONS = frozenset(
    {"correct", "conflict", "steered", "wrong_sign", "oracle_steered"}
)


def validate_expert_steering_conditions(
    conditions: tuple[str, ...],
) -> tuple[ExpertSteeringCondition, ...]:
    """Validate source-free expert-steering rollout conditions."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Expert-steering conditions must be nonempty and unique")
    unknown = set(conditions) - VALID_EXPERT_STEERING_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown expert-steering conditions: {sorted(unknown)}")
    return cast(tuple[ExpertSteeringCondition, ...], conditions)


@dataclass(frozen=True)
class ExpertSteeringEpisode:
    """One closed-loop outcome for a disjoint development scene."""

    episode_index: int
    condition: ExpertSteeringCondition
    success: bool
    steps: int
    steering_calls: int


@dataclass(frozen=True)
class ExpertSteeringEvaluationReport:
    """Closed-loop report with explicit source-free invariants."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    training_episode_indices: tuple[int, ...]
    episode_indices: tuple[int, ...]
    conditions: tuple[ExpertSteeringCondition, ...]
    spec: dict[str, object]
    fitted_scale: float
    noise_seed: int
    direction_noise_seed: int
    simulator_seed: int
    max_steps: int
    scene_condition: SceneCondition | None
    training_target_activations_used: int
    evaluation_target_activations_used: int
    steered_runtime_correct_prompt_forwards: int
    runtime_prompt_rewrites: int
    language_token_interventions: int
    episodes: tuple[ExpertSteeringEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""

        payload = cast(dict[str, object], asdict(self))
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def run_expert_steering_closed_loop(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    training_episode_indices: tuple[int, ...],
    episode_indices: tuple[int, ...],
    conditions: tuple[ExpertSteeringCondition, ...] = ("steered",),
    expert_layer: int = 15,
    action_token_positions: tuple[int, ...] = (0,),
    expert_interface: Literal["post_residual", "expert_output"] = "post_residual",
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    direction_noise_seed: int = 107,
    simulator_seed: int = 42,
    max_steps: int = 220,
    scene_condition: SceneCondition | None = None,
    progress: Callable[[ExpertSteeringEpisode], None] | None = None,
) -> ExpertSteeringEvaluationReport:
    """Evaluate fixed training-scene expert directions in closed loop."""

    if not training_episode_indices or not episode_indices:
        raise ValueError("Training and evaluation episodes must be nonempty")
    if set(training_episode_indices) & set(episode_indices):
        raise ValueError("Training and evaluation episodes must be disjoint")
    if len(set(training_episode_indices)) != len(training_episode_indices):
        raise ValueError("Training episodes must be unique")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be unique")
    if max_steps <= 0:
        raise ValueError("Maximum rollout steps must be positive")
    if set((*training_episode_indices, *episode_indices)) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if scene_condition is not None:
        spatial_support_task(task_id)
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
    training_pairs = _initial_snapshots(
        adapter,
        policy,
        training_episode_indices,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        suite=suite,
        task_id=task_id,
        resolution=resolution,
        simulator_seed=simulator_seed,
        scene_condition=scene_condition,
    )
    scene_directions: dict[int, StepDirectionMap] = {}
    for episode_index, (conflict, correct) in training_pairs.items():
        scene_directions[episode_index] = capture_scene_expert_directions(
            policy,
            conflict,
            correct,
            fixed_noise(policy, direction_noise_seed + episode_index, torch_device),
            spec,
        )
    directions = mean_expert_directions(scene_directions)
    fitted_scale = fit_expert_direction_scale(
        policy,
        training_pairs,
        spec,
        directions,
        noise_seed=direction_noise_seed,
        device=torch_device,
    )
    calibrated = _scaled_directions(directions, fitted_scale)
    wrong_sign = _scaled_directions(calibrated, -1.0)
    outcomes: list[ExpertSteeringEpisode] = []
    for episode_index in episode_indices:
        for condition in conditions:
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
            step = 0
            success = False
            steering_calls = 0
            try:
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                if scene_condition is not None:
                    observation, _ = apply_unique_bowl_scene(
                        environment, condition=scene_condition
                    )
                while step < max_steps and not success:
                    prompt = correct_prompt if condition == "correct" else conflict_prompt
                    snapshot = prefix_snapshot(
                        policy, prepare_libero_batch(adapter, observation, prompt)
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    if condition in {"correct", "conflict"}:
                        chunk = sample_with_expert_directions(policy, snapshot, noise, None, None)
                    elif condition == "oracle_steered":
                        correct_snapshot = prefix_snapshot(
                            policy,
                            prepare_libero_batch(adapter, observation, correct_prompt),
                        )
                        chunk = sample_with_oracle_expert_patch(
                            policy, snapshot, correct_snapshot, noise, spec
                        )
                        steering_calls += 1
                    else:
                        active = calibrated if condition == "steered" else wrong_sign
                        chunk = sample_with_expert_directions(policy, snapshot, noise, spec, active)
                        steering_calls += 1
                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    if terminated or truncated:
                        break
            finally:
                environment.close()
            outcome = ExpertSteeringEpisode(
                episode_index=episode_index,
                condition=condition,
                success=success,
                steps=step,
                steering_calls=steering_calls,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(outcome)
    num_steps = int(policy.model.config.num_steps)
    return ExpertSteeringEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        training_episode_indices=training_episode_indices,
        episode_indices=episode_indices,
        conditions=conditions,
        spec=asdict(spec),
        fitted_scale=fitted_scale,
        noise_seed=noise_seed,
        direction_noise_seed=direction_noise_seed,
        simulator_seed=simulator_seed,
        max_steps=max_steps,
        scene_condition=scene_condition,
        training_target_activations_used=len(training_episode_indices) * num_steps,
        evaluation_target_activations_used=sum(
            episode.steering_calls * num_steps
            for episode in outcomes
            if episode.condition == "oracle_steered"
        ),
        steered_runtime_correct_prompt_forwards=sum(
            episode.steering_calls
            for episode in outcomes
            if episode.condition == "oracle_steered"
        ),
        runtime_prompt_rewrites=0,
        language_token_interventions=0,
        episodes=tuple(outcomes),
    )
