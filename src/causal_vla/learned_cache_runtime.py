"""Leave-one-scene-out cache steering at causally localized SmolVLA loci."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from causal_vla.causal_trace import KVCache, aggregate_patch_rows, recovery_score
from causal_vla.causal_trace_runtime import PrefixSnapshot, prefix_snapshot
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.residual_runtime import decode_action
from causal_vla.smoke import fixed_noise, prepare_libero_batch


@dataclass(frozen=True)
class CacheDeltaLocus:
    """One cache component, layer, token set, and denoising-time locus."""

    name: str
    layer: int
    token_positions: tuple[int, ...]
    component: str
    denoising_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Cache-delta locus name must not be empty")
        if self.layer < 0:
            raise ValueError("Cache-delta layer must be nonnegative")
        if not self.token_positions or len(set(self.token_positions)) != len(
            self.token_positions
        ):
            raise ValueError("Token positions must be nonempty and unique")
        if any(position < 0 for position in self.token_positions):
            raise ValueError("Token positions must be absolute and nonnegative")
        if self.component not in {"key_states", "value_states"}:
            raise ValueError("Cache-delta component must be key_states or value_states")
        if not self.denoising_steps or len(set(self.denoising_steps)) != len(
            self.denoising_steps
        ):
            raise ValueError("Denoising steps must be nonempty and unique")


@dataclass(frozen=True)
class LearnedCacheSpec:
    """A named composition of independently localized cache deltas."""

    name: str
    loci: tuple[CacheDeltaLocus, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.loci:
            raise ValueError("Learned-cache spec name and loci must not be empty")
        if len({locus.name for locus in self.loci}) != len(self.loci):
            raise ValueError("Cache-delta locus names must be unique within a spec")


@dataclass(frozen=True)
class LearnedCacheScreenReport:
    """Cross-scene generalization of localized activation deltas."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    episode_indices: tuple[int, ...]
    training_strategy: str
    training_episode_indices: tuple[int, ...]
    noise_seed: int
    simulator_seed: int
    direction_diagnostics: tuple[dict[str, object], ...]
    specs: tuple[dict[str, object], ...]
    rows: tuple[dict[str, object], ...]
    ranked_specs: tuple[dict[str, object], ...]
    heldout_target_activations_used: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""

        payload = cast(dict[str, object], asdict(self))
        payload["direction_diagnostics"] = list(self.direction_diagnostics)
        payload["specs"] = list(self.specs)
        payload["rows"] = list(self.rows)
        payload["ranked_specs"] = list(self.ranked_specs)
        return payload


LearnedCacheCondition = Literal[
    "correct",
    "conflict",
    "steered",
    "grasp_gated",
    "oracle_steered",
    "oracle_language_state",
    "oracle_all_prefix",
    "wrong_sign",
    "aligned_steered",
]
VALID_LEARNED_CACHE_CONDITIONS = frozenset(
    {
        "correct",
        "conflict",
        "steered",
        "grasp_gated",
        "oracle_steered",
        "oracle_language_state",
        "oracle_all_prefix",
        "wrong_sign",
        "aligned_steered",
    }
)


def validate_learned_cache_conditions(
    conditions: tuple[str, ...],
) -> tuple[LearnedCacheCondition, ...]:
    """Validate closed-loop learned-cache conditions."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Learned-cache conditions must be nonempty and unique")
    unknown = set(conditions) - VALID_LEARNED_CACHE_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown learned-cache conditions: {sorted(unknown)}")
    return cast(tuple[LearnedCacheCondition, ...], conditions)


@dataclass(frozen=True)
class LearnedCacheEpisode:
    """One closed-loop outcome for a held development scene."""

    episode_index: int
    training_episodes: tuple[int, ...]
    condition: LearnedCacheCondition
    success: bool
    steps: int
    steering_calls: int
    grasp_latched_step: int | None
    initial_gripper_width: float
    minimum_gripper_width: float


@dataclass(frozen=True)
class LearnedCacheEvaluationReport:
    """Closed-loop LOO report with runtime-leakage invariants."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    episode_indices: tuple[int, ...]
    conditions: tuple[LearnedCacheCondition, ...]
    spec: dict[str, object]
    direction_scale_by_episode: dict[str, float]
    noise_seed: int
    simulator_seed: int
    max_steps: int
    grasp_close_fraction: float
    training_correct_prompt_forwards: int
    steered_runtime_correct_prompt_forwards: int
    per_episode_heldout_target_activations_used: int
    runtime_prompt_rewrites: int
    language_token_interventions: int
    auxiliary_prompt_forwards: int
    episodes: tuple[LearnedCacheEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""

        payload = cast(dict[str, object], asdict(self))
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


DirectionMap = dict[tuple[int, str, tuple[int, ...]], Tensor]


def _direction_key(locus: CacheDeltaLocus) -> tuple[int, str, tuple[int, ...]]:
    return locus.layer, locus.component, locus.token_positions


def mean_cache_directions(
    training_pairs: tuple[tuple[PrefixSnapshot, PrefixSnapshot], ...],
    loci: tuple[CacheDeltaLocus, ...],
) -> DirectionMap:
    """Average correct-minus-conflict cache deltas over training scenes only."""

    if not training_pairs:
        raise ValueError("At least one training cache pair is required")
    directions: DirectionMap = {}
    for locus in loci:
        scene_deltas: list[Tensor] = []
        for conflict, correct in training_pairs:
            conflict_tensor = conflict.cache[locus.layer][locus.component]
            correct_tensor = correct.cache[locus.layer][locus.component]
            positions = list(locus.token_positions)
            scene_deltas.append(correct_tensor[:, positions] - conflict_tensor[:, positions])
        directions[_direction_key(locus)] = torch.stack(scene_deltas).mean(dim=0)
    return directions


def add_cache_directions(
    cache: KVCache,
    loci: tuple[CacheDeltaLocus, ...],
    directions: DirectionMap,
) -> KVCache:
    """Return a cache with selected learned deltas added, leaving input tensors untouched."""

    result: KVCache = dict(cache)
    copied_layers: set[int] = set()
    for locus in loci:
        key = _direction_key(locus)
        if key not in directions:
            raise ValueError(f"Missing learned direction for locus {locus.name}")
        if locus.layer not in copied_layers:
            result[locus.layer] = dict(cache[locus.layer])
            copied_layers.add(locus.layer)
        base = result[locus.layer][locus.component]
        updated = base.clone()
        positions = list(locus.token_positions)
        direction = directions[key].to(device=updated.device, dtype=updated.dtype)
        if updated[:, positions].shape != direction.shape:
            raise ValueError(f"Direction shape does not match locus {locus.name}")
        updated[:, positions] = updated[:, positions] + direction
        result[locus.layer][locus.component] = updated
    return result


def sample_with_learned_cache(
    policy: Any,
    snapshot: PrefixSnapshot,
    noise: Tensor,
    spec: LearnedCacheSpec | None,
    directions: DirectionMap | None,
) -> Tensor:
    """Integrate an action chunk with temporally localized learned cache deltas."""

    model = policy.model
    if (spec is None) != (directions is None):
        raise ValueError("Learned cache spec and directions must be provided together")
    x_t = noise.clone()
    dt = -1.0 / int(model.config.num_steps)
    cache_by_step: dict[int, KVCache] = {}
    if spec is not None and directions is not None:
        for step in range(int(model.config.num_steps)):
            active = tuple(locus for locus in spec.loci if step in locus.denoising_steps)
            if active:
                cache_by_step[step] = add_cache_directions(snapshot.cache, active, directions)
    with torch.inference_mode():
        for step in range(int(model.config.num_steps)):
            timestep = torch.full(
                (x_t.shape[0],),
                1.0 + step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks,
                past_key_values=cache_by_step.get(step, snapshot.cache),
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


def _direction_diagnostics(
    episode_pairs: dict[int, tuple[PrefixSnapshot, PrefixSnapshot]],
    loci: tuple[CacheDeltaLocus, ...],
) -> tuple[dict[str, object], ...]:
    diagnostics: list[dict[str, object]] = []
    for locus in loci:
        vectors: list[Tensor] = []
        for conflict, correct in episode_pairs.values():
            positions = list(locus.token_positions)
            delta = (
                correct.cache[locus.layer][locus.component][:, positions]
                - conflict.cache[locus.layer][locus.component][:, positions]
            )
            vectors.append(delta.float().reshape(-1).cpu())
        cosines: list[float] = []
        for left_index, left in enumerate(vectors):
            for right in vectors[left_index + 1 :]:
                cosines.append(float(torch.nn.functional.cosine_similarity(left, right, dim=0)))
        diagnostics.append(
            {
                "locus": locus.name,
                "layer": locus.layer,
                "component": locus.component,
                "token_positions": list(locus.token_positions),
                "denoising_steps": list(locus.denoising_steps),
                "pairwise_cosine_mean": sum(cosines) / len(cosines),
                "pairwise_cosine_min": min(cosines),
                "scene_delta_norms": [float(vector.norm()) for vector in vectors],
            }
        )
    return tuple(diagnostics)


def _dual_all_step_loci(num_steps: int) -> tuple[CacheDeltaLocus, CacheDeltaLocus]:
    all_steps = tuple(range(num_steps))
    return (
        CacheDeltaLocus("l7_state_value_all", 7, (143,), "value_states", all_steps),
        CacheDeltaLocus("l13_changed_key_all", 13, (135,), "key_states", all_steps),
    )


def _scaled_directions(directions: DirectionMap, scale: float) -> DirectionMap:
    return {key: value * scale for key, value in directions.items()}


def _gripper_width(observation: Mapping[str, Any]) -> float:
    """Return the absolute two-finger opening from a LIBERO observation."""

    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise TypeError("Observation has no robot_state mapping")
    gripper = robot_state.get("gripper")
    if not isinstance(gripper, Mapping):
        raise TypeError("Observation has no gripper mapping")
    qpos = gripper.get("qpos")
    if qpos is None:
        raise TypeError("Observation has no gripper qpos")
    values = tuple(float(value) for value in qpos)
    if len(values) != 2:
        raise ValueError("Expected a two-finger gripper state")
    return sum(abs(value) for value in values)


def fit_executed_action_scale(
    policy: Any,
    training_pairs: dict[int, tuple[PrefixSnapshot, PrefixSnapshot]],
    spec: LearnedCacheSpec,
    directions: DirectionMap,
    *,
    noise_seed: int,
    device: torch.device,
) -> float:
    """Fit one scalar by least squares on training-scene first actions."""

    if not training_pairs:
        raise ValueError("Action-space scale fitting requires training scenes")
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for episode_index, (conflict, correct) in training_pairs.items():
        noise = fixed_noise(policy, noise_seed + episode_index, device)
        conflict_chunk = sample_with_learned_cache(policy, conflict, noise, None, None)
        correct_chunk = sample_with_learned_cache(policy, correct, noise, None, None)
        steered_chunk = sample_with_learned_cache(
            policy, conflict, noise, spec, directions
        )
        target_delta = (
            correct_chunk[:, :1] - conflict_chunk[:, :1]
        ).detach().cpu().double()
        steering_effect = (
            steered_chunk[:, :1] - conflict_chunk[:, :1]
        ).detach().cpu().double()
        numerator += torch.sum(steering_effect * target_delta)
        denominator += torch.sum(steering_effect * steering_effect)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-20:
        raise ValueError("Training steering effects are too small to fit a scale")
    scale = float(numerator / denominator)
    if not math.isfinite(scale):
        raise ValueError("Fitted action-space steering scale is not finite")
    return scale


def run_learned_cache_loo_screen(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    episode_indices: tuple[int, ...],
    training_episode_indices: tuple[int, ...] | None = None,
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 107,
    simulator_seed: int = 42,
    progress: Callable[[int, str], None] | None = None,
) -> LearnedCacheScreenReport:
    """Test causal-locus cache directions with strict leave-one-scene-out fitting."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be nonempty and unique")
    if training_episode_indices is None:
        if len(episode_indices) < 3:
            raise ValueError("At least three episodes are required for leave-one-out")
        snapshot_episodes = episode_indices
        training_strategy = "leave_one_scene_out"
        reported_training_episodes: tuple[int, ...] = ()
    else:
        if not training_episode_indices or len(set(training_episode_indices)) != len(
            training_episode_indices
        ):
            raise ValueError("Training episodes must be nonempty and unique")
        if set(training_episode_indices) & set(episode_indices):
            raise ValueError("Fixed training and evaluation episodes must be disjoint")
        snapshot_episodes = (*training_episode_indices, *episode_indices)
        training_strategy = "fixed_disjoint_split"
        reported_training_episodes = training_episode_indices
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    all_steps = tuple(range(int(policy.model.config.num_steps)))
    state_value = CacheDeltaLocus("l7_state_value_t2", 7, (143,), "value_states", (2,))
    changed_key = CacheDeltaLocus("l13_changed_key_t9", 13, (135,), "key_states", (9,))
    state_value_all = CacheDeltaLocus(
        "l7_state_value_all", 7, (143,), "value_states", all_steps
    )
    changed_key_all = CacheDeltaLocus(
        "l13_changed_key_all", 13, (135,), "key_states", all_steps
    )
    specs = (
        LearnedCacheSpec("l7_state_value_local", (state_value,)),
        LearnedCacheSpec("l13_changed_key_local", (changed_key,)),
        LearnedCacheSpec("dual_locus_local", (state_value, changed_key)),
        LearnedCacheSpec("l7_state_value_all", (state_value_all,)),
        LearnedCacheSpec("l13_changed_key_all", (changed_key_all,)),
        LearnedCacheSpec("dual_locus_all", (state_value_all, changed_key_all)),
    )
    calibrated_spec = LearnedCacheSpec(
        "dual_locus_all_calibrated", (state_value_all, changed_key_all)
    )
    unique_loci = (state_value, changed_key)
    torch_device = torch.device(device)
    action_dim = int(policy.config.action_feature.shape[0])
    episode_pairs: dict[int, tuple[PrefixSnapshot, PrefixSnapshot]] = {}
    for episode_index in snapshot_episodes:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            if native_prompt != correct_prompt:
                raise ValueError("Correct prompt must exactly match the native instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            conflict = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, conflict_prompt)
            )
            correct = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, correct_prompt)
            )
            episode_pairs[episode_index] = (conflict, correct)
        finally:
            environment.close()

    rows: list[dict[str, object]] = []
    for heldout_episode in episode_indices:
        conflict, correct = episode_pairs[heldout_episode]
        training_episodes = (
            tuple(episode for episode in episode_indices if episode != heldout_episode)
            if training_episode_indices is None
            else training_episode_indices
        )
        training_pairs = tuple(episode_pairs[episode] for episode in training_episodes)
        directions = mean_cache_directions(training_pairs, unique_loci)
        training_pair_map = {
            episode: episode_pairs[episode] for episode in training_episodes
        }
        fitted_scale = fit_executed_action_scale(
            policy,
            training_pair_map,
            calibrated_spec,
            directions,
            noise_seed=noise_seed,
            device=torch_device,
        )
        noise = fixed_noise(policy, noise_seed + heldout_episode, torch_device)
        conflict_chunk = sample_with_learned_cache(policy, conflict, noise, None, None)
        correct_chunk = sample_with_learned_cache(policy, correct, noise, None, None)
        for spec in specs:
            steered_chunk = sample_with_learned_cache(policy, conflict, noise, spec, directions)
            executed = recovery_score(
                conflict_chunk[:, :1, :action_dim],
                correct_chunk[:, :1, :action_dim],
                steered_chunk[:, :1, :action_dim],
            )
            chunk = recovery_score(conflict_chunk, correct_chunk, steered_chunk)
            rows.append(
                {
                    "episode_index": heldout_episode,
                    "training_episodes": list(training_episodes),
                    "layer": -1,
                    "denoising_step": -1,
                    "token_patch": spec.name,
                    "components": "learned_cache_delta",
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
                progress(heldout_episode, spec.name)
        calibrated_chunk = sample_with_learned_cache(
            policy,
            conflict,
            noise,
            calibrated_spec,
            _scaled_directions(directions, fitted_scale),
        )
        calibrated_executed = recovery_score(
            conflict_chunk[:, :1, :action_dim],
            correct_chunk[:, :1, :action_dim],
            calibrated_chunk[:, :1, :action_dim],
        )
        calibrated_full_chunk = recovery_score(
            conflict_chunk, correct_chunk, calibrated_chunk
        )
        rows.append(
            {
                "episode_index": heldout_episode,
                "training_episodes": list(training_episodes),
                "direction_scale": fitted_scale,
                "layer": -1,
                "denoising_step": -1,
                "token_patch": calibrated_spec.name,
                "components": "learned_cache_delta",
                "executed_directional_recovery": calibrated_executed.directional_recovery,
                "executed_mse_recovery": calibrated_executed.mse_recovery,
                "executed_effect_l2": calibrated_executed.effect_l2,
                "executed_conflict_correct_l2": (
                    calibrated_executed.conflict_correct_l2
                ),
                "chunk_directional_recovery": (
                    calibrated_full_chunk.directional_recovery
                ),
                "chunk_mse_recovery": calibrated_full_chunk.mse_recovery,
                "chunk_effect_l2": calibrated_full_chunk.effect_l2,
                "chunk_conflict_correct_l2": (
                    calibrated_full_chunk.conflict_correct_l2
                ),
            }
        )
        if progress is not None:
            progress(heldout_episode, calibrated_spec.name)
    serialized_specs: tuple[dict[str, object], ...] = tuple(
        {
            "name": spec.name,
            "loci": [asdict(locus) for locus in spec.loci],
        }
        for spec in (*specs, calibrated_spec)
    )
    return LearnedCacheScreenReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        episode_indices=episode_indices,
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        training_strategy=training_strategy,
        training_episode_indices=reported_training_episodes,
        direction_diagnostics=_direction_diagnostics(
            {
                episode: episode_pairs[episode]
                for episode in (
                    episode_indices
                    if training_episode_indices is None
                    else training_episode_indices
                )
            },
            unique_loci,
        ),
        specs=serialized_specs,
        rows=tuple(rows),
        ranked_specs=tuple(aggregate_patch_rows(rows)),
        heldout_target_activations_used=0,
    )


def run_learned_cache_closed_loop(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    episode_indices: tuple[int, ...],
    conditions: tuple[LearnedCacheCondition, ...],
    training_episode_indices: tuple[int, ...] | None = None,
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    simulator_seed: int = 42,
    max_steps: int = 220,
    grasp_close_fraction: float = 0.75,
    progress: Callable[[LearnedCacheEpisode], None] | None = None,
) -> LearnedCacheEvaluationReport:
    """Evaluate the selected dual-locus direction with LOO scene isolation."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be nonempty and unique")
    if training_episode_indices is None:
        if len(episode_indices) < 3:
            raise ValueError("At least three episodes are required for leave-one-out")
        direction_episodes = episode_indices
    else:
        if not training_episode_indices or len(set(training_episode_indices)) != len(
            training_episode_indices
        ):
            raise ValueError("Training episodes must be nonempty and unique")
        if set(training_episode_indices) & set(episode_indices):
            raise ValueError("Fixed training and evaluation episodes must be disjoint")
        direction_episodes = training_episode_indices
    if max_steps <= 0:
        raise ValueError("Maximum rollout steps must be positive")
    if not 0.0 < grasp_close_fraction < 1.0:
        raise ValueError("Grasp-close fraction must lie strictly between zero and one")
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    torch_device = torch.device(device)
    loci = _dual_all_step_loci(int(policy.model.config.num_steps))
    spec = LearnedCacheSpec("dual_locus_all", loci)

    training_pairs: dict[int, tuple[PrefixSnapshot, PrefixSnapshot]] = {}
    for episode_index in direction_episodes:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            if native_prompt != correct_prompt:
                raise ValueError("Correct prompt must exactly match the native instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            conflict = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, conflict_prompt)
            )
            correct = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, correct_prompt)
            )
            training_pairs[episode_index] = (conflict, correct)
        finally:
            environment.close()

    outcomes: list[LearnedCacheEpisode] = []
    direction_scale_by_episode: dict[str, float] = {}
    runtime_correct_prompt_forwards = 0
    auxiliary_prompt_forwards = 0
    for episode_index in episode_indices:
        fold_training_episodes = (
            tuple(episode for episode in episode_indices if episode != episode_index)
            if training_episode_indices is None
            else training_episode_indices
        )
        directions = mean_cache_directions(
            tuple(training_pairs[episode] for episode in fold_training_episodes), loci
        )
        fitted_scale = fit_executed_action_scale(
            policy,
            {
                episode: training_pairs[episode] for episode in fold_training_episodes
            },
            spec,
            directions,
            noise_seed=107,
            device=torch_device,
        )
        direction_scale_by_episode[str(episode_index)] = fitted_scale
        calibrated_directions = _scaled_directions(directions, fitted_scale)
        for condition in conditions:
            environment, native_prompt, _ = adapter.create_env(
                task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            if native_prompt != correct_prompt:
                environment.close()
                raise ValueError("Correct prompt must exactly match the native instruction")
            step = 0
            success = False
            steering_calls = 0
            grasp_latched_step: int | None = None
            try:
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                initial_gripper_width = _gripper_width(observation)
                minimum_gripper_width = initial_gripper_width
                close_threshold = initial_gripper_width * grasp_close_fraction
                while step < max_steps and not success:
                    current_gripper_width = _gripper_width(observation)
                    minimum_gripper_width = min(
                        minimum_gripper_width, current_gripper_width
                    )
                    if (
                        condition == "grasp_gated"
                        and grasp_latched_step is None
                        and current_gripper_width <= close_threshold
                    ):
                        grasp_latched_step = step
                    prompt = (
                        correct_prompt
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
                    steering_active = not (
                        condition == "grasp_gated" and grasp_latched_step is not None
                    )
                    if condition in {"correct", "conflict"} or not steering_active:
                        chunk = sample_with_learned_cache(
                            policy, snapshot, noise, None, None
                        )
                    else:
                        active_spec = spec
                        if condition in {
                            "oracle_steered",
                            "oracle_language_state",
                            "oracle_all_prefix",
                        }:
                            correct_snapshot = prefix_snapshot(
                                policy,
                                prepare_libero_batch(
                                    adapter, observation, correct_prompt
                                ),
                            )
                            active_loci = loci
                            if condition in {
                                "oracle_language_state",
                                "oracle_all_prefix",
                            }:
                                language_start = sum(snapshot.image_lengths)
                                positions = (
                                    tuple(range(language_start, snapshot.sequence_length))
                                    if condition == "oracle_language_state"
                                    else tuple(range(snapshot.sequence_length))
                                )
                                broad_value = CacheDeltaLocus(
                                    f"l7_{condition}",
                                    7,
                                    positions,
                                    "value_states",
                                    tuple(range(int(policy.model.config.num_steps))),
                                )
                                active_loci = (broad_value, loci[1])
                                active_spec = LearnedCacheSpec(
                                    condition, active_loci
                                )
                            active_directions = mean_cache_directions(
                                ((snapshot, correct_snapshot),), active_loci
                            )
                            runtime_correct_prompt_forwards += 1
                            auxiliary_prompt_forwards += 1
                        else:
                            active_directions = (
                                _scaled_directions(calibrated_directions, -1.0)
                                if condition == "wrong_sign"
                                else calibrated_directions
                            )
                        chunk = sample_with_learned_cache(
                            policy, snapshot, noise, active_spec, active_directions
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
            outcome = LearnedCacheEpisode(
                episode_index=episode_index,
                training_episodes=fold_training_episodes,
                condition=condition,
                success=success,
                steps=step,
                steering_calls=steering_calls,
                grasp_latched_step=grasp_latched_step,
                initial_gripper_width=initial_gripper_width,
                minimum_gripper_width=minimum_gripper_width,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(outcome)
    return LearnedCacheEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        episode_indices=episode_indices,
        conditions=conditions,
        spec={"name": spec.name, "loci": [asdict(locus) for locus in loci]},
        direction_scale_by_episode=direction_scale_by_episode,
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        max_steps=max_steps,
        grasp_close_fraction=grasp_close_fraction,
        training_correct_prompt_forwards=len(direction_episodes),
        steered_runtime_correct_prompt_forwards=runtime_correct_prompt_forwards,
        per_episode_heldout_target_activations_used=0,
        runtime_prompt_rewrites=0,
        language_token_interventions=0,
        auxiliary_prompt_forwards=auxiliary_prompt_forwards,
        episodes=tuple(outcomes),
    )
