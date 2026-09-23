"""Schema-v2 runtime for sealed multi-noise monitor-steering replication.

This module is intentionally separate from ``monitored_knn_runtime``.  The v1
evaluation hash-locks that module's exact bytes, so extending it in place would
invalidate the already reported 96-rollout result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from causal_vla.causal_trace_runtime import prefix_snapshot
from causal_vla.expert_blend_runtime import (
    ExpertBlendBank,
    load_expert_blend_bank,
    sample_with_expert_blend,
)
from causal_vla.expert_knn_runtime import (
    ExpertPrototypeBank,
    _sample_natural,
    load_expert_prototype_bank,
    sample_with_expert_knn,
)
from causal_vla.expert_ridge_runtime import (
    ExpertRidgeBank,
    load_expert_ridge_bank,
    sample_with_expert_ridge,
)
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.learned_cache_runtime import _gripper_width
from causal_vla.location_monitor_runtime import (
    LocationProbeBank,
    assess_location_claim,
    load_location_probe_bank,
)
from causal_vla.monitored_knn_runtime import (
    DestinationSteeringSupervisor,
    GripperPhaseRelease,
    ReleaseMode,
    oracle_conflict_assessment,
    oracle_presence_conflict_assessment,
)
from causal_vla.object_intervention import apply_hidden_object_scene, object_conflict_task
from causal_vla.object_monitor_runtime import (
    PresenceProbeBank,
    assess_presence_claim,
    load_presence_probe_bank,
)
from causal_vla.residual_runtime import decode_action
from causal_vla.routing import TemporalConflictGate
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch

ConfirmatoryCondition = Literal[
    "conflict",
    "always_steer",
    "wrong_sign",
    "random",
    "oracle_monitor",
    "learned_monitor",
    "aligned_no_steer",
    "aligned_monitor",
    "aligned_forced_steer",
]
TriggerSource = Literal["none", "learned", "oracle", "forced"]

VALID_CONFIRMATORY_CONDITIONS = frozenset(
    {
        "conflict",
        "always_steer",
        "wrong_sign",
        "random",
        "oracle_monitor",
        "learned_monitor",
        "aligned_no_steer",
        "aligned_monitor",
        "aligned_forced_steer",
    }
)
ALIGNED_CONDITIONS = frozenset({"aligned_no_steer", "aligned_monitor", "aligned_forced_steer"})
ROUTED_STEERING_CONDITIONS = frozenset(
    {
        "wrong_sign",
        "random",
        "oracle_monitor",
        "learned_monitor",
        "aligned_monitor",
        "aligned_forced_steer",
    }
)


def validate_confirmatory_conditions(
    conditions: tuple[str, ...],
) -> tuple[ConfirmatoryCondition, ...]:
    """Validate a unique, nonempty v2 condition list."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Confirmatory conditions must be nonempty and unique")
    unknown = set(conditions) - VALID_CONFIRMATORY_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown confirmatory conditions: {sorted(unknown)}")
    return cast(tuple[ConfirmatoryCondition, ...], conditions)


def is_aligned_condition(condition: ConfirmatoryCondition) -> bool:
    """Return whether the action prompt and scene must be aligned."""

    return condition in ALIGNED_CONDITIONS


def uses_routed_steering(condition: ConfirmatoryCondition) -> bool:
    """Return whether steering follows the temporal gate or release supervisor."""

    return condition in ROUTED_STEERING_CONDITIONS


def location_claimed_label(
    *,
    spatial_mode: bool,
    condition: ConfirmatoryCondition,
    claimed_source_label: str,
) -> str:
    """Return the language-side location label used by a location monitor.

    Spatial aligned trials retain the task's instructed support.  ``table`` is
    the aligned source only for the legacy goal-suite formulation; it is not a
    member of the spatial support-monitor vocabulary.
    """

    if spatial_mode or condition != "aligned_monitor":
        return claimed_source_label
    return "table"


def _eef_height(observation: Mapping[str, Any]) -> float:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise TypeError("Observation has no robot_state mapping")
    eef = robot_state.get("eef")
    if not isinstance(eef, Mapping):
        raise TypeError("Observation has no end-effector state")
    position = eef.get("pos")
    if position is None:
        raise TypeError("Observation has no end-effector position")
    values = tuple(float(value) for value in position)
    if len(values) != 3:
        raise ValueError("Expected a three-dimensional end-effector position")
    return values[2]


@dataclass(frozen=True)
class ConfirmatoryEpisode:
    episode_index: int
    condition: ConfirmatoryCondition
    success: bool
    steps: int
    steering_calls: int
    monitor_checks: int
    monitor_status: str | None
    monitor_visual_label: str | None
    monitor_visual_confidence: float | None
    release_monitor_checks: int
    release_visual_label: str | None
    release_visual_confidence: float | None
    trigger_source: TriggerSource
    trigger_step: int | None
    grasp_latched_step: int | None
    release_step: int | None
    release_steps: tuple[int, ...]
    reactivation_steps: tuple[int, ...]
    initial_gripper_width: float
    minimum_gripper_width: float
    initial_eef_height: float
    maximum_eef_height: float
    grasp_latched_height: float | None
    scene_condition: SceneCondition | None
    prompt: str
    native_prompt: str


@dataclass(frozen=True)
class ConfirmatoryEvaluationReport:
    schema_version: int
    checkpoint: str
    steering_bank_path: str
    steering_controller_type: str
    monitor_bank_path: str
    monitor_type: str
    release_monitor_bank_path: str
    suite: str
    task_id: int
    episode_indices: tuple[int, ...]
    conditions: tuple[ConfirmatoryCondition, ...]
    correct_prompt: str
    conflict_prompt: str
    aligned_prompt: str
    noise_seed: int
    noise_schedule_id: str
    simulator_seed: int
    random_seed: int
    max_steps: int
    grasp_close_fraction: float
    grasp_reopen_fraction: float
    grasp_lift_threshold: float
    grasp_release_patience: int
    release_reactivation_patience: int
    release_mode: ReleaseMode
    release_monitor_interval: int
    monitor_arm_timeout: int
    evaluation_target_activations_used: int
    runtime_counterfactual_prompt_forwards: int
    runtime_prompt_rewrites: int
    language_token_interventions: int
    scene_interventions: bool
    hidden_object: str | None
    instructed_support: str | None
    alternate_support: str | None
    episodes: tuple[ConfirmatoryEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def _sample_steered(
    policy: Any,
    snapshot: Any,
    noise: torch.Tensor,
    bank: ExpertPrototypeBank | ExpertRidgeBank | ExpertBlendBank,
    *,
    condition: ConfirmatoryCondition,
    environment_step: int,
    episode_index: int,
    random_seed: int,
) -> torch.Tensor:
    direction_sign = -1.0 if condition == "wrong_sign" else 1.0
    call_seed = (
        random_seed + episode_index * 10_000 + environment_step * 10
        if condition == "random"
        else None
    )
    if isinstance(bank, ExpertBlendBank):
        return sample_with_expert_blend(
            policy,
            snapshot,
            noise,
            bank,
            environment_step=environment_step,
            direction_sign=direction_sign,
            random_seed=call_seed,
        )
    if isinstance(bank, ExpertRidgeBank):
        return sample_with_expert_ridge(
            policy,
            snapshot,
            noise,
            bank,
            direction_sign=direction_sign,
            random_seed=call_seed,
        )
    return sample_with_expert_knn(
        policy,
        snapshot,
        noise,
        bank,
        environment_step=environment_step,
        direction_sign=direction_sign,
        random_seed=call_seed,
    )


def run_confirmatory_monitor_evaluation(
    checkpoint: str | Path,
    steering_bank_path: str | Path,
    monitor_bank_path: str | Path,
    *,
    release_monitor_bank_path: str | Path | None = None,
    task_id: int,
    episode_indices: tuple[int, ...],
    conditions: tuple[ConfirmatoryCondition, ...],
    suite: str,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int,
    simulator_seed: int = 42,
    random_seed: int = 30_007,
    max_steps: int = 220,
    grasp_close_fraction: float = 0.75,
    grasp_reopen_fraction: float = 0.90,
    grasp_lift_threshold: float = 0.03,
    grasp_release_patience: int = 2,
    release_reactivation_patience: int = 2,
    release_mode: ReleaseMode = "post_grasp_reopen",
    release_monitor_interval: int = 5,
    monitor_arm_timeout: int = 3,
    progress: Callable[[ConfirmatoryEpisode], None] | None = None,
) -> ConfirmatoryEvaluationReport:
    """Run the sealed v2 matrix, including a forced false-trigger stress test."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be nonempty and unique")
    if (
        not 0 < grasp_close_fraction < grasp_reopen_fraction <= 1
        or grasp_lift_threshold <= 0
        or grasp_release_patience <= 0
        or release_monitor_interval <= 0
        or release_reactivation_patience <= 0
    ):
        raise ValueError("Grasp release settings are invalid")

    spatial_mode = suite == "libero_spatial"
    object_mode = suite == "libero_object"
    if spatial_mode:
        spatial_spec = spatial_support_task(task_id)
        object_spec = None
        object_name = "black bowl"
        claimed_source_label = spatial_spec.instructed_support
        observed_conflict_label = spatial_spec.alternate_support
        destination_label = "plate"
        correct_prompt = spatial_spec.alternate_prompt
        conflict_prompt = spatial_spec.native_prompt
        aligned_prompt = spatial_spec.native_prompt
    elif object_mode:
        spatial_spec = None
        object_spec = object_conflict_task(task_id)
        object_name = object_spec.hidden_object
        claimed_source_label = "present"
        observed_conflict_label = "absent"
        destination_label = "present"
        correct_prompt = object_spec.native_prompt
        conflict_prompt = object_spec.conflict_prompt
        aligned_prompt = object_spec.native_prompt
        if release_monitor_bank_path is not None:
            raise ValueError("Object-presence mode does not use a release-monitor bank")
        if release_mode in {"destination_monitor", "destination_and_reopen"}:
            raise ValueError("Object-presence mode requires a gripper phase release")
    else:
        raise ValueError("Confirmatory replication supports object and spatial suites only")

    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)

    steering_header = torch.load(Path(steering_bank_path), map_location="cpu", weights_only=True)
    if not isinstance(steering_header, dict):
        raise ValueError("Steering bank payload is malformed")
    steering_bank: ExpertPrototypeBank | ExpertRidgeBank | ExpertBlendBank
    controller_type = steering_header.get("controller_type")
    if controller_type == "ridge":
        steering_bank = load_expert_ridge_bank(steering_bank_path)
        steering_controller_type = "ridge"
    elif controller_type == "knn":
        steering_bank = load_expert_prototype_bank(steering_bank_path)
        steering_controller_type = "knn"
    elif controller_type == "blend":
        steering_bank = load_expert_blend_bank(steering_bank_path)
        steering_controller_type = "blend"
    else:
        raise ValueError(f"Unsupported steering controller type {controller_type!r}")

    monitor_bank: LocationProbeBank | PresenceProbeBank
    release_monitor_bank: LocationProbeBank | None
    if object_mode:
        monitor_bank = load_presence_probe_bank(monitor_bank_path)
        release_monitor_path = Path(monitor_bank_path)
        release_monitor_bank = None
        monitor_type = "object_presence_probe"
    else:
        monitor_bank = load_location_probe_bank(monitor_bank_path)
        release_monitor_path = (
            Path(monitor_bank_path)
            if release_monitor_bank_path is None
            else Path(release_monitor_bank_path)
        )
        release_monitor_bank = load_location_probe_bank(release_monitor_path)
        monitor_type = "object_location_probe"
        if (
            release_mode in {"destination_monitor", "destination_and_reopen"}
            and destination_label not in release_monitor_bank.labels
        ):
            raise ValueError("Location monitor bank does not include the destination label")

    torch_device = torch.device(device)
    outcomes: list[ConfirmatoryEpisode] = []
    for episode_index in episode_indices:
        for condition in conditions:
            environment, native_prompt, _ = adapter.create_env(
                task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            expected_native_prompt = aligned_prompt if spatial_mode else correct_prompt
            if native_prompt.casefold() != expected_native_prompt.casefold():
                environment.close()
                raise ValueError("Native instruction differs from monitor task spec")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            episode_scene_condition: SceneCondition | None = None
            if spatial_mode:
                episode_scene_condition = (
                    "aligned" if is_aligned_condition(condition) else "conflict"
                )
                observation, _ = apply_unique_bowl_scene(
                    environment,
                    condition=episode_scene_condition,
                )
            elif object_mode:
                assert object_spec is not None
                observation, _ = apply_hidden_object_scene(
                    environment,
                    object_name=object_spec.hidden_object,
                )

            initial_gripper_width = _gripper_width(observation)
            minimum_gripper_width = initial_gripper_width
            initial_eef_height = _eef_height(observation)
            maximum_eef_height = initial_eef_height
            phase_release = GripperPhaseRelease(
                initial_gripper_width,
                close_fraction=grasp_close_fraction,
                reopen_fraction=grasp_reopen_fraction,
                patience=grasp_release_patience,
                mode=release_mode,
                lift_threshold=grasp_lift_threshold,
            )
            gate = TemporalConflictGate(
                trigger_patience=1,
                release_patience=grasp_release_patience,
                arm_timeout=monitor_arm_timeout,
            )
            destination_supervisor = DestinationSteeringSupervisor(
                release_patience=grasp_release_patience,
                reactivate_patience=release_reactivation_patience,
            )
            trigger_source: TriggerSource = "none"
            trigger_step: int | None = None
            if condition == "aligned_forced_steer":
                forced_assessment = (
                    oracle_presence_conflict_assessment()
                    if object_mode
                    else oracle_conflict_assessment(
                        cast(LocationProbeBank, monitor_bank).labels,
                        claimed_source_label,
                        observed_conflict_label,
                    )
                )
                forced_decision = gate.update(forced_assessment)
                if forced_decision.event != "trigger":
                    raise AssertionError("Forced false trigger did not activate the gate")
                trigger_source = "forced"
                trigger_step = 0
                if release_mode in {"destination_monitor", "destination_and_reopen"}:
                    destination_supervisor.arm()

            step = 0
            success = False
            steering_calls = 0
            monitor_checks = 0
            monitor_status: str | None = (
                "forced_conflict" if condition == "aligned_forced_steer" else None
            )
            monitor_visual_label: str | None = None
            monitor_visual_confidence: float | None = None
            release_monitor_checks = 0
            release_visual_label: str | None = None
            release_visual_confidence: float | None = None
            release_step: int | None = None
            try:
                while step < max_steps and not success:
                    current_gripper_width = _gripper_width(observation)
                    minimum_gripper_width = min(minimum_gripper_width, current_gripper_width)
                    maximum_eef_height = max(maximum_eef_height, _eef_height(observation))

                    if uses_routed_steering(condition) and gate.state == "armed":
                        if object_mode:
                            assert object_spec is not None
                            assert isinstance(monitor_bank, PresenceProbeBank)
                            monitored_object = (
                                object_spec.target_object
                                if condition == "aligned_monitor"
                                else object_spec.hidden_object
                            )
                            assessment = (
                                oracle_presence_conflict_assessment()
                                if condition == "oracle_monitor"
                                else assess_presence_claim(
                                    policy,
                                    adapter,
                                    observation,
                                    object_name=monitored_object,
                                    bank=monitor_bank,
                                )
                            )
                        else:
                            assert isinstance(monitor_bank, LocationProbeBank)
                            claimed_label = location_claimed_label(
                                spatial_mode=spatial_mode,
                                condition=condition,
                                claimed_source_label=claimed_source_label,
                            )
                            assessment = (
                                oracle_conflict_assessment(
                                    monitor_bank.labels,
                                    claimed_label,
                                    observed_conflict_label,
                                )
                                if condition == "oracle_monitor"
                                else assess_location_claim(
                                    policy,
                                    adapter,
                                    observation,
                                    object_name=object_name,
                                    claimed_label=claimed_label,
                                    bank=monitor_bank,
                                )
                            )
                        monitor_checks += 1
                        monitor_status = assessment.status
                        monitor_visual_label = assessment.vision_label
                        monitor_visual_confidence = assessment.vision_confidence
                        decision = gate.update(assessment)
                        if decision.event == "trigger":
                            trigger_source = (
                                "oracle" if condition == "oracle_monitor" else "learned"
                            )
                            trigger_step = step
                            if release_mode in {
                                "destination_monitor",
                                "destination_and_reopen",
                            }:
                                destination_supervisor.arm()

                    prompt = aligned_prompt if is_aligned_condition(condition) else conflict_prompt
                    snapshot = prefix_snapshot(
                        policy,
                        prepare_libero_batch(adapter, observation, prompt),
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    monitored_steering_active = gate.state == "active"
                    if (
                        release_mode in {"destination_monitor", "destination_and_reopen"}
                        and destination_supervisor.state != "waiting"
                    ):
                        monitored_steering_active = destination_supervisor.steering_active
                    steering_active = condition == "always_steer" or (
                        uses_routed_steering(condition) and monitored_steering_active
                    )
                    if steering_active:
                        chunk = _sample_steered(
                            policy,
                            snapshot,
                            noise,
                            steering_bank,
                            condition=condition,
                            environment_step=step,
                            episode_index=episode_index,
                            random_seed=random_seed,
                        )
                        steering_calls += 1
                    else:
                        chunk = _sample_natural(policy, snapshot, noise)

                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    post_action_gripper_width = _gripper_width(observation)
                    post_action_eef_height = _eef_height(observation)
                    minimum_gripper_width = min(minimum_gripper_width, post_action_gripper_width)
                    maximum_eef_height = max(maximum_eef_height, post_action_eef_height)
                    if gate.state == "active" and phase_release.update(
                        post_action_gripper_width,
                        step,
                        eef_height=post_action_eef_height,
                    ):
                        gate.release_for_phase_transition(
                            "post_grasp_reopen"
                            if release_mode == "post_grasp_reopen"
                            else "grasp_lifted"
                            if release_mode == "grasp_lift"
                            else "grasp_detected"
                        )
                        release_step = step

                    release_requires_reopen = release_mode == "destination_and_reopen"
                    release_sensor_ready = (
                        not release_requires_reopen
                        or post_action_gripper_width
                        >= initial_gripper_width * grasp_reopen_fraction
                    )
                    if (
                        release_mode in {"destination_monitor", "destination_and_reopen"}
                        and destination_supervisor.state in {"active", "released"}
                        and phase_release.grasp_latched_step is not None
                        and release_sensor_ready
                        and (
                            step % release_monitor_interval == 0
                            or release_requires_reopen
                            or success
                        )
                    ):
                        assert release_monitor_bank is not None
                        release_assessment = assess_location_claim(
                            policy,
                            adapter,
                            observation,
                            object_name=object_name,
                            claimed_label=destination_label,
                            bank=release_monitor_bank,
                        )
                        release_monitor_checks += 1
                        release_visual_label = release_assessment.vision_label
                        release_visual_confidence = release_assessment.vision_confidence
                        destination_aligned = (
                            release_assessment.status == "aligned"
                            and release_assessment.vision_label == destination_label
                        )
                        supervisor_event = destination_supervisor.update(
                            destination_aligned=destination_aligned,
                            step=step,
                        )
                        if supervisor_event == "release":
                            release_step = release_step if release_step is not None else step
                            phase_release.release_step = step
                    if terminated or truncated:
                        break
            finally:
                environment.close()

            outcome = ConfirmatoryEpisode(
                episode_index=episode_index,
                condition=condition,
                success=success,
                steps=step,
                steering_calls=steering_calls,
                monitor_checks=monitor_checks,
                monitor_status=monitor_status,
                monitor_visual_label=monitor_visual_label,
                monitor_visual_confidence=monitor_visual_confidence,
                release_monitor_checks=release_monitor_checks,
                release_visual_label=release_visual_label,
                release_visual_confidence=release_visual_confidence,
                trigger_source=trigger_source,
                trigger_step=trigger_step,
                grasp_latched_step=phase_release.grasp_latched_step,
                release_step=release_step,
                release_steps=tuple(destination_supervisor.release_steps or ()),
                reactivation_steps=tuple(destination_supervisor.reactivation_steps or ()),
                initial_gripper_width=initial_gripper_width,
                minimum_gripper_width=minimum_gripper_width,
                initial_eef_height=initial_eef_height,
                maximum_eef_height=maximum_eef_height,
                grasp_latched_height=phase_release.grasp_latched_height,
                scene_condition=episode_scene_condition,
                prompt=prompt,
                native_prompt=native_prompt,
            )
            outcomes.append(outcome)
            if progress is not None:
                progress(outcome)

    return ConfirmatoryEvaluationReport(
        schema_version=2,
        checkpoint=checkpoint_path,
        steering_bank_path=str(Path(steering_bank_path).resolve()),
        steering_controller_type=steering_controller_type,
        monitor_bank_path=str(Path(monitor_bank_path).resolve()),
        monitor_type=monitor_type,
        release_monitor_bank_path=str(release_monitor_path.resolve()),
        suite=suite,
        task_id=task_id,
        episode_indices=episode_indices,
        conditions=conditions,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        aligned_prompt=aligned_prompt,
        noise_seed=noise_seed,
        noise_schedule_id="base_plus_episode_10000_plus_step",
        simulator_seed=simulator_seed,
        random_seed=random_seed,
        max_steps=max_steps,
        grasp_close_fraction=grasp_close_fraction,
        grasp_reopen_fraction=grasp_reopen_fraction,
        grasp_lift_threshold=grasp_lift_threshold,
        grasp_release_patience=grasp_release_patience,
        release_reactivation_patience=release_reactivation_patience,
        release_mode=release_mode,
        release_monitor_interval=release_monitor_interval,
        monitor_arm_timeout=monitor_arm_timeout,
        evaluation_target_activations_used=0,
        runtime_counterfactual_prompt_forwards=0,
        runtime_prompt_rewrites=0,
        language_token_interventions=0,
        scene_interventions=True,
        hidden_object=object_spec.hidden_object if object_spec is not None else None,
        instructed_support=(spatial_spec.instructed_support if spatial_spec is not None else None),
        alternate_support=(spatial_spec.alternate_support if spatial_spec is not None else None),
        episodes=tuple(outcomes),
    )


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.confirmatory_monitor_runtime")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steering-bank", type=Path, required=True)
    parser.add_argument("--monitor-bank", type=Path, required=True)
    parser.add_argument("--release-monitor-bank", type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=_integers, required=True)
    parser.add_argument("--suite", choices=("libero_spatial", "libero_object"), required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--simulator-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=30_007)
    parser.add_argument("--grasp-close-fraction", type=float, default=0.75)
    parser.add_argument("--grasp-reopen-fraction", type=float, default=0.90)
    parser.add_argument("--grasp-lift-threshold", type=float, default=0.03)
    parser.add_argument("--grasp-release-patience", type=int, default=2)
    parser.add_argument("--release-reactivation-patience", type=int, default=2)
    parser.add_argument(
        "--release-mode",
        choices=(
            "grasp_close",
            "grasp_lift",
            "post_grasp_reopen",
            "destination_monitor",
            "destination_and_reopen",
        ),
        default="post_grasp_reopen",
    )
    parser.add_argument("--release-monitor-interval", type=int, default=5)
    parser.add_argument("--monitor-arm-timeout", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conditions = validate_confirmatory_conditions(
        tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    )
    report = run_confirmatory_monitor_evaluation(
        args.checkpoint,
        args.steering_bank,
        args.monitor_bank,
        release_monitor_bank_path=args.release_monitor_bank,
        task_id=args.task_id,
        episode_indices=args.episodes,
        conditions=conditions,
        suite=args.suite,
        device=args.device,
        noise_seed=args.noise_seed,
        simulator_seed=args.simulator_seed,
        random_seed=args.random_seed,
        grasp_close_fraction=args.grasp_close_fraction,
        grasp_reopen_fraction=args.grasp_reopen_fraction,
        grasp_lift_threshold=args.grasp_lift_threshold,
        grasp_release_patience=args.grasp_release_patience,
        release_reactivation_patience=args.release_reactivation_patience,
        release_mode=args.release_mode,
        release_monitor_interval=args.release_monitor_interval,
        monitor_arm_timeout=args.monitor_arm_timeout,
        progress=lambda outcome: print(
            f"[confirmatory-monitor] episode={outcome.episode_index} "
            f"condition={outcome.condition} success={outcome.success} "
            f"steps={outcome.steps} steering_calls={outcome.steering_calls} "
            f"trigger={outcome.trigger_source}@{outcome.trigger_step} "
            f"release={outcome.release_step}",
            file=sys.stderr,
            flush=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
