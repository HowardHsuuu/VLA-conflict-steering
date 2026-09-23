"""Closed-loop location monitoring and causally localized expert KNN steering."""

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
from causal_vla.claim_monitor_runtime import source_claim_specs
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
from causal_vla.object_intervention import apply_hidden_object_scene, object_conflict_task
from causal_vla.object_monitor_runtime import (
    PresenceProbeBank,
    assess_presence_claim,
    load_presence_probe_bank,
)
from causal_vla.residual_runtime import decode_action
from causal_vla.routing import (
    ConflictAssessment,
    ConflictDetector,
    EvidenceDistribution,
    TemporalConflictGate,
)
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch

MonitoredKNNCondition = Literal[
    "correct",
    "conflict",
    "always_steer",
    "wrong_sign",
    "random",
    "oracle_monitor",
    "learned_monitor",
    "aligned_no_steer",
    "aligned_monitor",
]
ReleaseMode = Literal[
    "grasp_close",
    "grasp_lift",
    "post_grasp_reopen",
    "destination_monitor",
    "destination_and_reopen",
]
DestinationSupervisorState = Literal["waiting", "active", "released"]
DestinationSupervisorEvent = Literal["none", "release", "reactivate"]
VALID_MONITORED_KNN_CONDITIONS = frozenset(
    {
        "correct",
        "conflict",
        "always_steer",
        "wrong_sign",
        "random",
        "oracle_monitor",
        "learned_monitor",
        "aligned_no_steer",
        "aligned_monitor",
    }
)


def validate_monitored_knn_conditions(
    conditions: tuple[str, ...],
) -> tuple[MonitoredKNNCondition, ...]:
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Monitored KNN conditions must be nonempty and unique")
    unknown = set(conditions) - VALID_MONITORED_KNN_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown monitored KNN conditions: {sorted(unknown)}")
    return cast(tuple[MonitoredKNNCondition, ...], conditions)


def condition_uses_monitor_steering(condition: MonitoredKNNCondition) -> bool:
    """Return whether an active monitor gate must route through steering."""

    return condition in {
        "wrong_sign",
        "random",
        "oracle_monitor",
        "learned_monitor",
        "aligned_monitor",
    }


def oracle_conflict_assessment(
    labels: tuple[str, ...],
    claimed_label: str,
    observed_label: str = "table",
) -> ConflictAssessment:
    """Construct an auditable oracle assessment from two explicit locations."""

    if (
        observed_label not in labels
        or claimed_label not in labels
        or claimed_label == observed_label
    ):
        raise ValueError("Oracle assessment requires distinct known locations")
    remainder = 0.01 / (len(labels) - 1)
    vision = EvidenceDistribution(
        "vision",
        "current_object_location",
        labels,
        tuple(0.99 if label == observed_label else remainder for label in labels),
    )
    language = EvidenceDistribution(
        "language",
        "current_object_location",
        labels,
        tuple(0.99 if label == claimed_label else remainder for label in labels),
    )
    assessment = ConflictDetector(divergence_threshold=0.25, confidence_floor=0.90).assess(
        vision, language
    )
    if assessment.status != "conflict":
        raise AssertionError("Oracle conflict construction did not trigger")
    return assessment


def oracle_presence_conflict_assessment() -> ConflictAssessment:
    """Construct an auditable oracle for a claimed object that is absent."""

    labels = ("absent", "present")
    vision = EvidenceDistribution("vision", "claimed_object_presence", labels, (0.99, 0.01))
    language = EvidenceDistribution("language", "claimed_object_presence", labels, (0.01, 0.99))
    assessment = ConflictDetector(divergence_threshold=0.25, confidence_floor=0.90).assess(
        vision, language
    )
    if assessment.status != "conflict":
        raise AssertionError("Oracle presence-conflict construction did not trigger")
    return assessment


@dataclass
class GripperPhaseRelease:
    """Detect acquisition or post-transport release from onboard gripper state."""

    initial_width: float
    close_fraction: float = 0.75
    reopen_fraction: float = 0.90
    patience: int = 2
    mode: ReleaseMode = "post_grasp_reopen"
    lift_threshold: float = 0.03
    grasp_latched_step: int | None = None
    grasp_latched_height: float | None = None
    release_step: int | None = None
    phase_streak: int = 0

    def __post_init__(self) -> None:
        if self.initial_width <= 0:
            raise ValueError("Initial gripper width must be positive")
        if not 0 < self.close_fraction < self.reopen_fraction <= 1:
            raise ValueError("Gripper close/reopen fractions are invalid")
        if self.patience <= 0:
            raise ValueError("Gripper release patience must be positive")
        if self.lift_threshold <= 0:
            raise ValueError("Grasp-lift threshold must be positive")

    def update(self, width: float, step: int, *, eef_height: float | None = None) -> bool:
        """Return true exactly once when the configured phase boundary is reached."""

        if width < 0 or step < 0:
            raise ValueError("Gripper observations must be nonnegative")
        if self.mode == "grasp_lift" and eef_height is None:
            raise ValueError("Grasp-lift release requires end-effector height")
        if self.release_step is not None:
            return False
        if self.grasp_latched_step is None:
            if width <= self.initial_width * self.close_fraction:
                self.phase_streak += 1
                if self.phase_streak >= self.patience:
                    self.grasp_latched_step = step
                    self.grasp_latched_height = eef_height
                    self.phase_streak = 0
                    if self.mode == "grasp_close":
                        self.release_step = step
                        return True
            else:
                self.phase_streak = 0
            return False
        if self.mode == "grasp_lift":
            if (
                self.grasp_latched_height is not None
                and eef_height is not None
                and width <= self.initial_width * self.reopen_fraction
                and eef_height >= self.grasp_latched_height + self.lift_threshold
            ):
                self.phase_streak += 1
                if self.phase_streak >= self.patience:
                    self.release_step = step
                    return True
            else:
                self.phase_streak = 0
        elif self.mode == "post_grasp_reopen":
            if width >= self.initial_width * self.reopen_fraction:
                self.phase_streak += 1
                if self.phase_streak >= self.patience:
                    self.release_step = step
                    return True
            else:
                self.phase_streak = 0
        return False


@dataclass
class DestinationSteeringSupervisor:
    """Reversible hysteresis for monitor-controlled destination release.

    A destination prediction can provisionally suspend steering, but it is not an
    irreversible decision.  If subsequent visual evidence no longer supports the
    destination claim, the supervisor reactivates steering after a separate patience
    window.  This prevents one transient probe error from controlling the rest of a
    rollout.
    """

    release_patience: int = 2
    reactivate_patience: int = 2
    state: DestinationSupervisorState = "waiting"
    aligned_streak: int = 0
    contradiction_streak: int = 0
    release_steps: list[int] | None = None
    reactivation_steps: list[int] | None = None

    def __post_init__(self) -> None:
        if self.release_patience <= 0 or self.reactivate_patience <= 0:
            raise ValueError("Destination supervisor patience must be positive")
        if self.release_steps is None:
            self.release_steps = []
        if self.reactivation_steps is None:
            self.reactivation_steps = []

    @property
    def steering_active(self) -> bool:
        return self.state == "active"

    def arm(self) -> None:
        """Activate supervision once the source-conflict gate triggers."""

        if self.state == "waiting":
            self.state = "active"

    def update(self, *, destination_aligned: bool, step: int) -> DestinationSupervisorEvent:
        """Update destination evidence and return any steering state transition."""

        if step < 0:
            raise ValueError("Destination supervisor step must be nonnegative")
        if self.state == "waiting":
            raise RuntimeError("Destination supervisor must be armed before update")
        if self.state == "active":
            self.contradiction_streak = 0
            self.aligned_streak = self.aligned_streak + 1 if destination_aligned else 0
            if self.aligned_streak >= self.release_patience:
                self.state = "released"
                self.aligned_streak = 0
                assert self.release_steps is not None
                self.release_steps.append(step)
                return "release"
            return "none"
        self.aligned_streak = 0
        self.contradiction_streak = 0 if destination_aligned else self.contradiction_streak + 1
        if self.contradiction_streak >= self.reactivate_patience:
            self.state = "active"
            self.contradiction_streak = 0
            assert self.reactivation_steps is not None
            self.reactivation_steps.append(step)
            return "reactivate"
        return "none"


def _eef_height(observation: Mapping[str, Any]) -> float:
    """Return end-effector height from the policy's onboard robot state."""

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
class MonitoredKNNEpisode:
    episode_index: int
    condition: MonitoredKNNCondition
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
class MonitoredKNNEvaluationReport:
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
    conditions: tuple[MonitoredKNNCondition, ...]
    correct_prompt: str
    conflict_prompt: str
    aligned_prompt: str
    noise_seed: int
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
    steered_runtime_correct_prompt_forwards: int
    runtime_prompt_rewrites: int
    language_token_interventions: int
    scene_interventions: bool
    hidden_object: str | None
    instructed_support: str | None
    alternate_support: str | None
    episodes: tuple[MonitoredKNNEpisode, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def run_monitored_knn_evaluation(
    checkpoint: str | Path,
    steering_bank_path: str | Path,
    monitor_bank_path: str | Path,
    *,
    release_monitor_bank_path: str | Path | None = None,
    task_id: int,
    episode_indices: tuple[int, ...],
    conditions: tuple[MonitoredKNNCondition, ...] = (
        "conflict",
        "always_steer",
        "oracle_monitor",
        "learned_monitor",
        "aligned_no_steer",
        "aligned_monitor",
    ),
    suite: str = "libero_goal",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 57,
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
    progress: Callable[[MonitoredKNNEpisode], None] | None = None,
) -> MonitoredKNNEvaluationReport:
    """Compare always-on, oracle, and learned monitoring in matched rollouts."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation episodes must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
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
        spatial_spec = None
        object_spec = None
        specs = {spec.task_id: spec for spec in source_claim_specs()}
        if task_id not in specs:
            raise ValueError(f"No source-claim spec for task {task_id}")
        spec = specs[task_id]
        object_name = spec.object_name
        claimed_source_label = spec.false_source_label
        observed_conflict_label = "table"
        destination_label = spec.destination_label
        correct_prompt = spec.correct_prompt
        conflict_prompt = spec.conflict_prompt
        aligned_prompt = spec.aligned_prompt
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
    outcomes: list[MonitoredKNNEpisode] = []
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
                    "aligned"
                    if condition in {"aligned_no_steer", "aligned_monitor"}
                    else "conflict"
                )
                observation, _ = apply_unique_bowl_scene(
                    environment, condition=episode_scene_condition
                )
            elif object_mode:
                assert object_spec is not None
                observation, _ = apply_hidden_object_scene(
                    environment, object_name=object_spec.hidden_object
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
            step = 0
            success = False
            steering_calls = 0
            monitor_checks = 0
            monitor_status: str | None = None
            monitor_visual_label: str | None = None
            monitor_visual_confidence: float | None = None
            release_monitor_checks = 0
            release_visual_label: str | None = None
            release_visual_confidence: float | None = None
            release_alignment_streak = 0
            trigger_step: int | None = None
            release_step: int | None = None
            gate = TemporalConflictGate(
                trigger_patience=1,
                release_patience=grasp_release_patience,
                arm_timeout=monitor_arm_timeout,
            )
            destination_supervisor = DestinationSteeringSupervisor(
                release_patience=grasp_release_patience,
                reactivate_patience=release_reactivation_patience,
            )
            try:
                while step < max_steps and not success:
                    current_gripper_width = _gripper_width(observation)
                    minimum_gripper_width = min(minimum_gripper_width, current_gripper_width)
                    maximum_eef_height = max(maximum_eef_height, _eef_height(observation))
                    if condition_uses_monitor_steering(condition) and gate.state == "armed":
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
                            claimed_label = (
                                claimed_source_label
                                if spatial_mode or condition != "aligned_monitor"
                                else "table"
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
                            trigger_step = step
                            if release_mode in {
                                "destination_monitor",
                                "destination_and_reopen",
                            }:
                                destination_supervisor.arm()
                    prompt = (
                        correct_prompt
                        if condition == "correct"
                        else aligned_prompt
                        if condition in {"aligned_no_steer", "aligned_monitor"}
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
                    monitored_steering_active = gate.state == "active"
                    if (
                        release_mode
                        in {
                            "destination_monitor",
                            "destination_and_reopen",
                        }
                        and destination_supervisor.state != "waiting"
                    ):
                        monitored_steering_active = destination_supervisor.steering_active
                    steering_active = condition == "always_steer" or (
                        condition_uses_monitor_steering(condition) and monitored_steering_active
                    )
                    if steering_active:
                        if isinstance(steering_bank, ExpertBlendBank):
                            chunk = sample_with_expert_blend(
                                policy,
                                snapshot,
                                noise,
                                steering_bank,
                                environment_step=step,
                                direction_sign=-1.0 if condition == "wrong_sign" else 1.0,
                                random_seed=(
                                    random_seed + episode_index * 10_000 + step * 10
                                    if condition == "random"
                                    else None
                                ),
                            )
                        elif isinstance(steering_bank, ExpertRidgeBank):
                            chunk = sample_with_expert_ridge(
                                policy,
                                snapshot,
                                noise,
                                steering_bank,
                                direction_sign=-1.0 if condition == "wrong_sign" else 1.0,
                                random_seed=(
                                    random_seed + episode_index * 10_000 + step * 10
                                    if condition == "random"
                                    else None
                                ),
                            )
                        else:
                            chunk = sample_with_expert_knn(
                                policy,
                                snapshot,
                                noise,
                                steering_bank,
                                environment_step=step,
                                direction_sign=-1.0 if condition == "wrong_sign" else 1.0,
                                random_seed=(
                                    random_seed + episode_index * 10_000 + step * 10
                                    if condition == "random"
                                    else None
                                ),
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
                        release_alignment_streak = (
                            release_alignment_streak + 1 if destination_aligned else 0
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
            outcome = MonitoredKNNEpisode(
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
    return MonitoredKNNEvaluationReport(
        schema_version=1,
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
        steered_runtime_correct_prompt_forwards=0,
        runtime_prompt_rewrites=0,
        language_token_interventions=0,
        scene_interventions=spatial_mode or object_mode,
        hidden_object=(object_spec.hidden_object if object_spec is not None else None),
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
    parser = argparse.ArgumentParser(prog="python -m causal_vla.monitored_knn_runtime")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steering-bank", type=Path, required=True)
    parser.add_argument("--monitor-bank", type=Path, required=True)
    parser.add_argument("--release-monitor-bank", type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=_integers, required=True)
    parser.add_argument(
        "--suite",
        choices=("libero_goal", "libero_spatial", "libero_object"),
        default="libero_goal",
    )
    parser.add_argument(
        "--conditions",
        default=(
            "conflict,always_steer,oracle_monitor,learned_monitor,aligned_no_steer,aligned_monitor"
        ),
    )
    parser.add_argument("--noise-seed", type=int, default=57)
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
    conditions = validate_monitored_knn_conditions(
        tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    )
    report = run_monitored_knn_evaluation(
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
            f"[monitored-knn] episode={outcome.episode_index} "
            f"condition={outcome.condition} success={outcome.success} "
            f"steps={outcome.steps} steering_calls={outcome.steering_calls} "
            f"trigger={outcome.trigger_step} release={outcome.release_step}",
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
