"""Frozen-VLM location monitor trained on counterbalanced robot trajectories."""

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
from torch import Tensor

from causal_vla.causal_trace_runtime import prefix_snapshot
from causal_vla.claim_monitor_runtime import SourceClaimSpec, source_claim_specs
from causal_vla.expert_knn_runtime import _sample_natural
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.interventions import InputActivationCapture
from causal_vla.learned_cache_runtime import _gripper_width
from causal_vla.monitor_calibration import (
    LabeledMonitorExample,
    MonitorCalibration,
    calibrate_conflict_monitor,
)
from causal_vla.probe_monitor_runtime import (
    ProbeDataset,
    fit_ridge_probe,
    leave_episode_out_logits,
    probe_logits,
)
from causal_vla.residual_runtime import decode_action, vlm_layers
from causal_vla.routing import ConflictAssessment, ConflictDetector, EvidenceDistribution
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_destination_scene,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch

_PROPOSITION = "current_object_location"
_LOCATION_LABELS = ("table", "cabinet", "stove", "rack")
_SPATIAL_LOCATION_LABELS = ("cabinet", "cookie_box", "plate", "ramekin", "stove")
_SPATIAL_TRAJECTORY_LOCATION_LABELS = (
    "cabinet",
    "cookie_box",
    "in_hand",
    "plate",
    "ramekin",
    "stove",
)
SpatialLocationTrainingCondition = Literal[
    "aligned",
    "conflict",
    "destination",
    "destination_rollout",
    "in_hand_rollout",
]


def language_location_evidence(labels: tuple[str, ...], claimed_label: str) -> EvidenceDistribution:
    """Encode one explicit instruction claim over a closed location vocabulary."""

    if claimed_label not in labels:
        raise ValueError("Claimed location is outside the monitor vocabulary")
    remainder = 0.02 / (len(labels) - 1)
    probabilities = tuple(0.98 if label == claimed_label else remainder for label in labels)
    return EvidenceDistribution("language", _PROPOSITION, labels, probabilities)


def location_vision_evidence(
    logits: Tensor, *, labels: tuple[str, ...], temperature: float
) -> EvidenceDistribution:
    """Convert one probe score vector into calibrated visual evidence."""

    if logits.shape != (len(labels),) or temperature <= 0:
        raise ValueError("Location logits or temperature are invalid")
    probabilities = torch.softmax(logits.double() / temperature, dim=0)
    return EvidenceDistribution(
        "vision",
        _PROPOSITION,
        labels,
        tuple(float(value) for value in probabilities),
    )


def capture_location_representations(
    policy: Any,
    batch: dict[str, Any],
    *,
    representation_layers: tuple[int, ...],
    representation_position: int,
) -> dict[int, Tensor]:
    """Capture several VLM residuals in one location-neutral prefix forward."""

    layers, _ = vlm_layers(policy)
    if not representation_layers or len(set(representation_layers)) != len(representation_layers):
        raise ValueError("Representation layers must be nonempty and unique")
    if min(representation_layers) < 0 or max(representation_layers) >= len(layers):
        raise ValueError("A representation layer is outside the VLM decoder")
    captures = {
        layer: InputActivationCapture(dtype=torch.float32) for layer in representation_layers
    }
    handles = [
        layers[layer].input_layernorm.register_forward_pre_hook(captures[layer])
        for layer in representation_layers
    ]
    try:
        prefix_snapshot(policy, batch)
    finally:
        for handle in handles:
            handle.remove()
    result: dict[int, Tensor] = {}
    for layer, capture in captures.items():
        if len(capture.records) != 1:
            raise RuntimeError("Expected exactly one location representation capture")
        residual = capture.records[0]
        if not -residual.shape[1] <= representation_position < residual.shape[1]:
            raise ValueError("Representation position is outside the prefix")
        result[layer] = residual[..., representation_position, :].reshape(1, -1).cpu()
    return result


@dataclass(frozen=True)
class LocationTrajectory:
    task_id: int
    episode_index: int
    success: bool
    steps: int
    final_label: str


@dataclass(frozen=True)
class LocationProbeBank:
    """Tensor-only linear location monitor and frozen detector thresholds."""

    labels: tuple[str, ...]
    representation_layer: int
    representation_position: int
    weight: Tensor
    alpha: float
    temperature: float
    divergence_threshold: float
    confidence_floor: float
    training_task_ids: tuple[int, ...]
    training_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[1] != len(self.labels):
            raise ValueError("Location probe weight has the wrong shape")
        if self.weight.shape[0] < 2 or self.alpha <= 0 or self.temperature <= 0:
            raise ValueError("Location probe hyperparameters are invalid")
        if not 0 <= self.divergence_threshold <= 1 or not 0 <= self.confidence_floor <= 1:
            raise ValueError("Location monitor thresholds must lie in [0, 1]")


def save_location_probe_bank(bank: LocationProbeBank, path: str | Path) -> None:
    """Save a weights-only compatible monitor artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema_version": 1, **asdict(bank)}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_location_probe_bank(path: str | Path) -> LocationProbeBank:
    """Load and validate a location monitor without arbitrary pickle globals."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported location probe-bank schema")
    return LocationProbeBank(
        labels=tuple(str(value) for value in payload["labels"]),
        representation_layer=int(payload["representation_layer"]),
        representation_position=int(payload["representation_position"]),
        weight=cast(Tensor, payload["weight"]),
        alpha=float(payload["alpha"]),
        temperature=float(payload["temperature"]),
        divergence_threshold=float(payload["divergence_threshold"]),
        confidence_floor=float(payload["confidence_floor"]),
        training_task_ids=tuple(int(value) for value in payload["training_task_ids"]),
        training_episode_indices=tuple(int(value) for value in payload["training_episode_indices"]),
    )


def _monitor_examples(
    dataset: ProbeDataset, logits: Tensor, *, temperature: float
) -> tuple[LabeledMonitorExample, ...]:
    examples: list[LabeledMonitorExample] = []
    for row in range(dataset.features.shape[0]):
        true_label = dataset.labels[int(dataset.label_indices[row])]
        vision = location_vision_evidence(
            logits[row], labels=dataset.labels, temperature=temperature
        )
        examples.append(
            LabeledMonitorExample(
                task_id=int(dataset.task_ids[row]),
                episode_index=int(dataset.episode_indices[row]),
                conflict_type="counterbalanced_object_location",
                expected_status="aligned",
                true_visual_label=true_label,
                vision=vision,
                language=language_location_evidence(dataset.labels, true_label),
            )
        )
        for claimed_label in dataset.labels:
            if claimed_label == true_label:
                continue
            examples.append(
                LabeledMonitorExample(
                    task_id=int(dataset.task_ids[row]),
                    episode_index=int(dataset.episode_indices[row]),
                    conflict_type="counterbalanced_object_location",
                    expected_status="conflict",
                    true_visual_label=true_label,
                    vision=vision,
                    language=language_location_evidence(dataset.labels, claimed_label),
                )
            )
    return tuple(examples)


@dataclass(frozen=True)
class LocationSelectionRow:
    representation_layer: int
    alpha: float
    temperature: float
    multiclass_accuracy: float
    calibration: MonitorCalibration


@dataclass(frozen=True)
class LocationProbeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    representation_layers: tuple[int, ...]
    representation_position: int
    labels: tuple[str, ...]
    trajectories: tuple[LocationTrajectory, ...]
    selected_layer: int
    selected_alpha: float
    selected_temperature: float
    calibration: MonitorCalibration
    selection_rows: tuple[LocationSelectionRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"calibration", "selection_rows", "trajectories"}
            },
            "trajectories": [asdict(item) for item in self.trajectories],
            "calibration": self.calibration.to_dict(),
            "selection_rows": [
                {
                    "representation_layer": row.representation_layer,
                    "alpha": row.alpha,
                    "temperature": row.temperature,
                    "multiclass_accuracy": row.multiclass_accuracy,
                    "selected_operating_point": asdict(row.calibration.selected),
                }
                for row in self.selection_rows
            ],
        }


@dataclass(frozen=True)
class SpatialLocationState:
    task_id: int
    episode_index: int
    condition: SpatialLocationTrainingCondition
    location_label: str


@dataclass(frozen=True)
class SpatialLocationProbeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    representation_layers: tuple[int, ...]
    representation_position: int
    labels: tuple[str, ...]
    states: tuple[SpatialLocationState, ...]
    destination_trajectories: tuple[LocationTrajectory, ...]
    noise_seed: int
    max_steps: int
    in_hand_sample_stride: int
    in_hand_min_steps_after_grasp: int
    grasp_close_fraction: float
    grasp_patience: int
    selected_layer: int
    selected_alpha: float
    selected_temperature: float
    calibration: MonitorCalibration
    selection_rows: tuple[LocationSelectionRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"calibration", "selection_rows", "states"}
            },
            "states": [asdict(item) for item in self.states],
            "calibration": self.calibration.to_dict(),
            "selection_rows": [
                {
                    "representation_layer": row.representation_layer,
                    "alpha": row.alpha,
                    "temperature": row.temperature,
                    "multiclass_accuracy": row.multiclass_accuracy,
                    "selected_operating_point": asdict(row.calibration.selected),
                }
                for row in self.selection_rows
            ],
        }


def select_location_probe(
    datasets: Mapping[int, ProbeDataset],
    *,
    representation_position: int,
    alpha_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    temperature_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> tuple[LocationProbeBank, tuple[LocationSelectionRow, ...], MonitorCalibration]:
    """Select layer and calibration by leave-episode-out development predictions."""

    if not datasets or not alpha_candidates or not temperature_candidates:
        raise ValueError("Location-probe selection candidates must be nonempty")
    rows: list[LocationSelectionRow] = []
    for layer, dataset in datasets.items():
        for alpha in alpha_candidates:
            logits = leave_episode_out_logits(dataset, alpha=alpha)
            accuracy = float((logits.argmax(dim=1) == dataset.label_indices).double().mean())
            for temperature in temperature_candidates:
                calibration = calibrate_conflict_monitor(
                    _monitor_examples(dataset, logits, temperature=temperature),
                    divergence_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
                    confidence_candidates=(
                        0.40,
                        0.45,
                        0.50,
                        0.55,
                        0.60,
                        0.65,
                        0.70,
                        0.75,
                        0.80,
                        0.85,
                        0.90,
                    ),
                    max_aligned_false_trigger_rate=0.0,
                )
                rows.append(LocationSelectionRow(layer, alpha, temperature, accuracy, calibration))
    selected = max(
        rows,
        key=lambda row: (
            -row.calibration.selected.aligned_false_trigger_rate,
            row.calibration.selected.conflict_recall,
            row.multiclass_accuracy,
            -row.calibration.selected.aligned_abstention_rate,
            -row.alpha,
            -row.temperature,
            -row.representation_layer,
        ),
    )
    dataset = datasets[selected.representation_layer]
    bank = LocationProbeBank(
        labels=dataset.labels,
        representation_layer=selected.representation_layer,
        representation_position=representation_position,
        weight=fit_ridge_probe(
            dataset.features,
            dataset.label_indices,
            classes=len(dataset.labels),
            alpha=selected.alpha,
        ),
        alpha=selected.alpha,
        temperature=selected.temperature,
        divergence_threshold=selected.calibration.selected.divergence_threshold,
        confidence_floor=selected.calibration.selected.confidence_floor,
        training_task_ids=tuple(sorted(set(dataset.task_ids.tolist()))),
        training_episode_indices=tuple(sorted(set(dataset.episode_indices.tolist()))),
    )
    return bank, tuple(rows), selected.calibration


def fit_location_probe_bank(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (1, 2, 4, 9),
    episode_indices: tuple[int, ...] = tuple(range(6)),
    representation_layers: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 31),
    representation_position: int = -2,
    suite: str = "libero_goal",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 7,
    simulator_seed: int = 42,
    max_steps: int = 220,
    progress: Callable[[LocationTrajectory], None] | None = None,
) -> tuple[LocationProbeBank, LocationProbeFitReport]:
    """Collect initial/final states and fit a counterbalanced location monitor."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    specs_by_task = {spec.task_id: spec for spec in source_claim_specs()}
    try:
        specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    except KeyError as error:
        raise ValueError(f"No source-claim spec for task {error.args[0]}") from error
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    torch_device = torch.device(device)
    feature_lists: dict[int, list[Tensor]] = {layer: [] for layer in representation_layers}
    label_indices: list[int] = []
    row_tasks: list[int] = []
    row_episodes: list[int] = []
    trajectories: list[LocationTrajectory] = []

    def append_state(
        observation: dict[str, Any], spec: SourceClaimSpec, label: str, episode: int
    ) -> None:
        batch = prepare_libero_batch(adapter, observation, spec.location_question)
        captured = capture_location_representations(
            policy,
            batch,
            representation_layers=representation_layers,
            representation_position=representation_position,
        )
        for layer, feature in captured.items():
            feature_lists[layer].append(feature)
        label_indices.append(_LOCATION_LABELS.index(label))
        row_tasks.append(spec.task_id)
        row_episodes.append(episode)

    for spec in specs:
        for episode_index in episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                spec.task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            if native_prompt.casefold() != spec.correct_prompt.casefold():
                environment.close()
                raise ValueError("Native instruction differs from source-claim spec")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            append_state(observation, spec, "table", episode_index)
            step = 0
            success = False
            try:
                while step < max_steps and not success:
                    snapshot = prefix_snapshot(
                        policy,
                        prepare_libero_batch(adapter, observation, spec.correct_prompt),
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    chunk = _sample_natural(policy, snapshot, noise)
                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    if terminated or truncated:
                        break
                if success:
                    append_state(observation, spec, spec.destination_label, episode_index)
            finally:
                environment.close()
            outcome = LocationTrajectory(
                spec.task_id,
                episode_index,
                success,
                step,
                spec.destination_label,
            )
            trajectories.append(outcome)
            if progress is not None:
                progress(outcome)
    episodes_by_label = {
        label: {
            episode
            for label_index, episode in zip(label_indices, row_episodes, strict=True)
            if _LOCATION_LABELS[label_index] == label
        }
        for label in _LOCATION_LABELS
    }
    underrepresented = {
        label: sorted(episodes)
        for label, episodes in episodes_by_label.items()
        if len(episodes) < 2
    }
    if underrepresented:
        raise RuntimeError(
            "Location classes need successful states from at least two episode indices: "
            f"{underrepresented}"
        )
    datasets = {
        layer: ProbeDataset(
            features=torch.cat(features),
            label_indices=torch.tensor(label_indices, dtype=torch.int64),
            task_ids=torch.tensor(row_tasks, dtype=torch.int64),
            episode_indices=torch.tensor(row_episodes, dtype=torch.int64),
            labels=_LOCATION_LABELS,
        )
        for layer, features in feature_lists.items()
    }
    bank, selection_rows, calibration = select_location_probe(
        datasets, representation_position=representation_position
    )
    report = LocationProbeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        representation_layers=representation_layers,
        representation_position=representation_position,
        labels=_LOCATION_LABELS,
        trajectories=tuple(trajectories),
        selected_layer=bank.representation_layer,
        selected_alpha=bank.alpha,
        selected_temperature=bank.temperature,
        calibration=calibration,
        selection_rows=selection_rows,
    )
    return bank, report


def fit_spatial_location_probe_bank(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = (1, 2, 3, 5),
    representation_layers: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 31),
    representation_position: int = -2,
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 67,
    simulator_seed: int = 42,
    max_steps: int = 220,
    destination_rollout_task_ids: tuple[int, ...] = (),
    destination_rollout_episode_indices: tuple[int, ...] = (),
    in_hand_sample_stride: int = 10,
    in_hand_min_steps_after_grasp: int = 5,
    grasp_close_fraction: float = 0.75,
    grasp_patience: int = 2,
    progress: Callable[[SpatialLocationState], None] | None = None,
    trajectory_progress: Callable[[LocationTrajectory], None] | None = None,
) -> tuple[LocationProbeBank, SpatialLocationProbeFitReport]:
    """Fit a counterbalanced support monitor from visual-only scene pairs."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if bool(destination_rollout_task_ids) != bool(destination_rollout_episode_indices):
        raise ValueError("Destination rollout tasks and episodes must be specified together")
    if set(destination_rollout_episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if (
        max_steps <= 0
        or in_hand_sample_stride <= 0
        or in_hand_min_steps_after_grasp < 0
        or not 0 < grasp_close_fraction < 1
        or grasp_patience <= 0
    ):
        raise ValueError("Spatial trajectory collection settings are invalid")
    tasks = tuple(spatial_support_task(task_id) for task_id in task_ids)
    rollout_tasks = tuple(
        spatial_support_task(task_id) for task_id in destination_rollout_task_ids
    )
    if not set(destination_rollout_task_ids) <= set(task_ids):
        raise ValueError("Destination rollout tasks must be included in spatial fit tasks")
    spatial_labels = (
        _SPATIAL_TRAJECTORY_LOCATION_LABELS
        if rollout_tasks
        else _SPATIAL_LOCATION_LABELS
    )
    observed_labels = {
        label
        for task in tasks
        for label in (task.instructed_support, task.alternate_support)
    } | {"plate"}
    if observed_labels != set(_SPATIAL_LOCATION_LABELS):
        raise ValueError(
            "Spatial monitor fitting requires coverage of cabinet, cookie_box, "
            "plate, ramekin, and stove"
        )
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    torch_device = torch.device(device)
    feature_lists: dict[int, list[Tensor]] = {
        layer: [] for layer in representation_layers
    }
    label_indices: list[int] = []
    row_tasks: list[int] = []
    row_episodes: list[int] = []
    states: list[SpatialLocationState] = []
    destination_trajectories: list[LocationTrajectory] = []

    def append_state(
        observation: dict[str, Any],
        *,
        task_id: int,
        episode_index: int,
        condition: SpatialLocationTrainingCondition,
        location_label: str,
    ) -> None:
        batch = prepare_libero_batch(
            adapter, observation, "Where is the black bowl right now?"
        )
        captured = capture_location_representations(
            policy,
            batch,
            representation_layers=representation_layers,
            representation_position=representation_position,
        )
        for layer, feature in captured.items():
            feature_lists[layer].append(feature)
        label_indices.append(spatial_labels.index(location_label))
        row_tasks.append(task_id)
        row_episodes.append(episode_index)
        state = SpatialLocationState(
            task_id, episode_index, condition, location_label
        )
        states.append(state)
        if progress is not None:
            progress(state)

    for task in tasks:
        for episode_index in episode_indices:
            for condition in cast(
                tuple[SpatialLocationTrainingCondition, ...],
                ("aligned", "conflict", "destination"),
            ):
                environment, native_prompt, _ = adapter.create_env(
                    task.task_id,
                    suite=suite,
                    resolution=resolution,
                    episode_index=episode_index,
                )
                try:
                    if native_prompt != task.native_prompt:
                        raise ValueError("Native instruction differs from intervention task")
                    environment.reset(seed=simulator_seed + episode_index)
                    observation, _ = (
                        apply_unique_bowl_destination_scene(environment)
                        if condition == "destination"
                        else apply_unique_bowl_scene(
                            environment,
                            condition=cast(SceneCondition, condition),
                        )
                    )
                finally:
                    environment.close()
                location_label = (
                    task.instructed_support
                    if condition == "aligned"
                    else task.alternate_support
                    if condition == "conflict"
                    else "plate"
                )
                append_state(
                    observation,
                    task_id=task.task_id,
                    episode_index=episode_index,
                    condition=condition,
                    location_label=location_label,
                )
    for task in rollout_tasks:
        for episode_index in destination_rollout_episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                task.task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            step = 0
            success = False
            in_hand_samples = 0
            try:
                if native_prompt != task.native_prompt:
                    raise ValueError("Native instruction differs from intervention task")
                environment.reset(seed=simulator_seed + episode_index)
                observation, _ = apply_unique_bowl_scene(
                    environment, condition="conflict"
                )
                initial_gripper_width = _gripper_width(observation)
                close_streak = 0
                grasp_latched_step: int | None = None
                while step < max_steps and not success:
                    snapshot = prefix_snapshot(
                        policy,
                        prepare_libero_batch(adapter, observation, task.alternate_prompt),
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + step,
                        torch_device,
                    )
                    chunk = _sample_natural(policy, snapshot, noise)
                    action = decode_action(adapter, chunk[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    step += 1
                    success = bool(info.get("is_success", False))
                    gripper_width = _gripper_width(observation)
                    if grasp_latched_step is None:
                        if gripper_width <= initial_gripper_width * grasp_close_fraction:
                            close_streak += 1
                            if close_streak >= grasp_patience:
                                grasp_latched_step = step
                        else:
                            close_streak = 0
                    elif (
                        not success
                        and gripper_width <= initial_gripper_width * 0.90
                        and step >= grasp_latched_step + in_hand_min_steps_after_grasp
                        and (
                            step - grasp_latched_step - in_hand_min_steps_after_grasp
                        )
                        % in_hand_sample_stride
                        == 0
                    ):
                        append_state(
                            observation,
                            task_id=task.task_id,
                            episode_index=episode_index,
                            condition="in_hand_rollout",
                            location_label="in_hand",
                        )
                        in_hand_samples += 1
                    if terminated or truncated:
                        break
                if success:
                    if in_hand_samples == 0:
                        raise RuntimeError(
                            "Successful destination rollout produced no in-hand samples"
                        )
                    append_state(
                        observation,
                        task_id=task.task_id,
                        episode_index=episode_index,
                        condition="destination_rollout",
                        location_label="plate",
                    )
            finally:
                environment.close()
            trajectory = LocationTrajectory(
                task.task_id,
                episode_index,
                success,
                step,
                "plate",
            )
            destination_trajectories.append(trajectory)
            if trajectory_progress is not None:
                trajectory_progress(trajectory)
    if destination_trajectories and not all(
        trajectory.success for trajectory in destination_trajectories
    ):
        failures = [
            (trajectory.task_id, trajectory.episode_index)
            for trajectory in destination_trajectories
            if not trajectory.success
        ]
        raise RuntimeError(f"Destination rollout collection failed for {failures}")
    datasets = {
        layer: ProbeDataset(
            features=torch.cat(features),
            label_indices=torch.tensor(label_indices, dtype=torch.int64),
            task_ids=torch.tensor(row_tasks, dtype=torch.int64),
            episode_indices=torch.tensor(row_episodes, dtype=torch.int64),
            labels=spatial_labels,
        )
        for layer, features in feature_lists.items()
    }
    bank, selection_rows, calibration = select_location_probe(
        datasets, representation_position=representation_position
    )
    report = SpatialLocationProbeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        representation_layers=representation_layers,
        representation_position=representation_position,
        labels=spatial_labels,
        states=tuple(states),
        destination_trajectories=tuple(destination_trajectories),
        noise_seed=noise_seed,
        max_steps=max_steps,
        in_hand_sample_stride=in_hand_sample_stride,
        in_hand_min_steps_after_grasp=in_hand_min_steps_after_grasp,
        grasp_close_fraction=grasp_close_fraction,
        grasp_patience=grasp_patience,
        selected_layer=bank.representation_layer,
        selected_alpha=bank.alpha,
        selected_temperature=bank.temperature,
        calibration=calibration,
        selection_rows=selection_rows,
    )
    return bank, report


def assess_location_claim(
    policy: Any,
    adapter: Any,
    observation: dict[str, Any],
    *,
    object_name: str,
    claimed_label: str,
    bank: LocationProbeBank,
) -> ConflictAssessment:
    """Assess one instruction claim without putting its location into the visual query."""

    vision = predict_location_evidence(
        policy,
        adapter,
        observation,
        object_name=object_name,
        bank=bank,
    )
    language = language_location_evidence(bank.labels, claimed_label)
    return ConflictDetector(
        divergence_threshold=bank.divergence_threshold,
        confidence_floor=bank.confidence_floor,
    ).assess(vision, language)


def predict_location_evidence(
    policy: Any,
    adapter: Any,
    observation: dict[str, Any],
    *,
    object_name: str,
    bank: LocationProbeBank,
) -> EvidenceDistribution:
    """Predict visual object location from a claim-independent query."""

    question = f"Where is the {object_name} right now?"
    batch = prepare_libero_batch(adapter, observation, question)
    representation = capture_location_representations(
        policy,
        batch,
        representation_layers=(bank.representation_layer,),
        representation_position=bank.representation_position,
    )[bank.representation_layer]
    return location_vision_evidence(
        probe_logits(representation, bank.weight)[0],
        labels=bank.labels,
        temperature=bank.temperature,
    )


@dataclass(frozen=True)
class LocationMonitorExample:
    task_id: int
    episode_index: int
    predicted_label: str
    predicted_confidence: float
    aligned_status: str
    conflict_status: str
    conflict_target_correct: bool


@dataclass(frozen=True)
class LocationMonitorEvaluationReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    examples: tuple[LocationMonitorExample, ...]
    aligned_false_trigger_rate: float
    conflict_recall: float

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["examples"] = [asdict(example) for example in self.examples]
        return payload


@dataclass(frozen=True)
class SpatialLocationMonitorExample:
    task_id: int
    episode_index: int
    condition: SpatialLocationTrainingCondition
    claimed_label: str
    actual_label: str
    predicted_label: str
    predicted_confidence: float
    monitor_status: str
    expected_status: str
    correct: bool


@dataclass(frozen=True)
class SpatialLocationMonitorReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    conditions: tuple[SpatialLocationTrainingCondition, ...]
    examples: tuple[SpatialLocationMonitorExample, ...]
    aligned_false_trigger_rate: float
    conflict_recall: float
    actual_location_accuracy: float

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["examples"] = [asdict(example) for example in self.examples]
        return payload


def evaluate_location_probe_initial_states(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    task_ids: tuple[int, ...] = (1, 2, 4, 9),
    episode_indices: tuple[int, ...] = (6, 7, 8, 9),
    suite: str = "libero_goal",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[LocationMonitorExample], None] | None = None,
) -> LocationMonitorEvaluationReport:
    """Evaluate a frozen location monitor on unseen initial scene states."""

    if not task_ids or not episode_indices:
        raise ValueError("Evaluation tasks and episodes must be nonempty")
    if len(set(task_ids)) != len(task_ids) or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Evaluation tasks and episodes must be unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    specs_by_task = {spec.task_id: spec for spec in source_claim_specs()}
    try:
        specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    except KeyError as error:
        raise ValueError(f"No source-claim spec for task {error.args[0]}") from error
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    bank = load_location_probe_bank(bank_path)
    detector = ConflictDetector(
        divergence_threshold=bank.divergence_threshold,
        confidence_floor=bank.confidence_floor,
    )
    examples: list[LocationMonitorExample] = []
    for spec in specs:
        for episode_index in episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                spec.task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            try:
                if native_prompt.casefold() != spec.correct_prompt.casefold():
                    raise ValueError("Native instruction differs from source-claim spec")
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                vision = predict_location_evidence(
                    policy,
                    adapter,
                    observation,
                    object_name=spec.object_name,
                    bank=bank,
                )
                aligned = detector.assess(vision, language_location_evidence(bank.labels, "table"))
                conflict = detector.assess(
                    vision,
                    language_location_evidence(bank.labels, spec.false_source_label),
                )
            finally:
                environment.close()
            example = LocationMonitorExample(
                task_id=spec.task_id,
                episode_index=episode_index,
                predicted_label=vision.top_label,
                predicted_confidence=vision.confidence,
                aligned_status=aligned.status,
                conflict_status=conflict.status,
                conflict_target_correct=(
                    conflict.status == "conflict" and conflict.vision_label == "table"
                ),
            )
            examples.append(example)
            if progress is not None:
                progress(example)
    aligned_false_triggers = sum(example.aligned_status == "conflict" for example in examples)
    correct_conflict_triggers = sum(example.conflict_target_correct for example in examples)
    return LocationMonitorEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        examples=tuple(examples),
        aligned_false_trigger_rate=aligned_false_triggers / len(examples),
        conflict_recall=correct_conflict_triggers / len(examples),
    )


def evaluate_location_probe_spatial_interventions(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    task_ids: tuple[int, ...] = (7, 9),
    episode_indices: tuple[int, ...] = (0, 4, 6, 8),
    conditions: tuple[SpatialLocationTrainingCondition, ...] = (
        "aligned",
        "conflict",
    ),
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[SpatialLocationMonitorExample], None] | None = None,
) -> SpatialLocationMonitorReport:
    """Evaluate a frozen monitor on prompt-invariant visual conflict scenes."""

    if not task_ids or not episode_indices or not conditions:
        raise ValueError("Tasks, episodes, and conditions must be nonempty")
    if any(
        len(set(values)) != len(values)
        for values in (task_ids, episode_indices, conditions)
    ):
        raise ValueError("Tasks, episodes, and conditions must be unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    tasks = tuple(spatial_support_task(task_id) for task_id in task_ids)
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    bank = load_location_probe_bank(bank_path)
    required_labels = {
        label
        for task in tasks
        for label in (task.instructed_support, task.alternate_support)
    }
    missing_labels = required_labels - set(bank.labels)
    if missing_labels:
        raise ValueError(f"Location probe lacks spatial labels: {sorted(missing_labels)}")
    detector = ConflictDetector(
        divergence_threshold=bank.divergence_threshold,
        confidence_floor=bank.confidence_floor,
    )
    examples: list[SpatialLocationMonitorExample] = []
    for task in tasks:
        for episode_index in episode_indices:
            for condition in conditions:
                environment, _, _ = adapter.create_env(
                    task.task_id,
                    suite=suite,
                    resolution=resolution,
                    episode_index=episode_index,
                )
                try:
                    environment.reset(seed=simulator_seed + episode_index)
                    observation, _ = (
                        apply_unique_bowl_destination_scene(environment)
                        if condition == "destination"
                        else apply_unique_bowl_scene(
                            environment,
                            condition=cast(SceneCondition, condition),
                        )
                    )
                    vision = predict_location_evidence(
                        policy,
                        adapter,
                        observation,
                        object_name="black bowl",
                        bank=bank,
                    )
                    assessment = detector.assess(
                        vision,
                        language_location_evidence(
                            bank.labels,
                            "plate"
                            if condition == "destination"
                            else task.instructed_support,
                        ),
                    )
                finally:
                    environment.close()
                actual_label = (
                    task.instructed_support
                    if condition == "aligned"
                    else task.alternate_support
                    if condition == "conflict"
                    else "plate"
                )
                expected_status = "conflict" if condition == "conflict" else "aligned"
                example = SpatialLocationMonitorExample(
                    task_id=task.task_id,
                    episode_index=episode_index,
                    condition=condition,
                    claimed_label=(
                        "plate" if condition == "destination" else task.instructed_support
                    ),
                    actual_label=actual_label,
                    predicted_label=vision.top_label,
                    predicted_confidence=vision.confidence,
                    monitor_status=assessment.status,
                    expected_status=expected_status,
                    correct=(
                        assessment.status == expected_status
                        and vision.top_label == actual_label
                    ),
                )
                examples.append(example)
                if progress is not None:
                    progress(example)
    aligned_examples = [example for example in examples if example.condition == "aligned"]
    conflict_examples = [example for example in examples if example.condition == "conflict"]
    aligned_false_triggers = sum(
        example.monitor_status == "conflict" for example in aligned_examples
    )
    conflict_triggers = sum(
        example.monitor_status == "conflict" and example.predicted_label == example.actual_label
        for example in conflict_examples
    )
    location_correct = sum(
        example.predicted_label == example.actual_label for example in examples
    )
    return SpatialLocationMonitorReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        conditions=conditions,
        examples=tuple(examples),
        aligned_false_trigger_rate=(
            aligned_false_triggers / len(aligned_examples) if aligned_examples else 0.0
        ),
        conflict_recall=(
            conflict_triggers / len(conflict_examples) if conflict_examples else 0.0
        ),
        actual_location_accuracy=location_correct / len(examples),
    )


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.location_monitor_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--tasks", type=_integers, default=(1, 2, 4, 9))
    fit.add_argument("--episodes", type=_integers, default=tuple(range(6)))
    fit.add_argument("--representation-layers", type=_integers, default=(8, 12, 16, 20, 24, 28, 31))
    fit.add_argument("--representation-position", type=int, default=-2)
    fit.add_argument("--noise-seed", type=int, default=7)
    fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    fit.add_argument("--bank-output", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    spatial_fit = subparsers.add_parser("fit-spatial")
    spatial_fit.add_argument("--checkpoint", type=Path, required=True)
    spatial_fit.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    spatial_fit.add_argument("--episodes", type=_integers, default=(1, 2, 3, 5))
    spatial_fit.add_argument(
        "--representation-layers",
        type=_integers,
        default=(8, 12, 16, 20, 24, 28, 31),
    )
    spatial_fit.add_argument("--representation-position", type=int, default=-2)
    spatial_fit.add_argument("--noise-seed", type=int, default=67)
    spatial_fit.add_argument("--max-steps", type=int, default=220)
    spatial_fit.add_argument("--destination-rollout-tasks", type=_integers)
    spatial_fit.add_argument("--destination-rollout-episodes", type=_integers)
    spatial_fit.add_argument("--in-hand-sample-stride", type=int, default=10)
    spatial_fit.add_argument("--in-hand-min-steps-after-grasp", type=int, default=5)
    spatial_fit.add_argument("--grasp-close-fraction", type=float, default=0.75)
    spatial_fit.add_argument("--grasp-patience", type=int, default=2)
    spatial_fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    spatial_fit.add_argument("--bank-output", type=Path, required=True)
    spatial_fit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--tasks", type=_integers, default=(1, 2, 4, 9))
    evaluate.add_argument("--episodes", type=_integers, default=(6, 7, 8, 9))
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    evaluate.add_argument("--output", type=Path, required=True)
    spatial = subparsers.add_parser("evaluate-spatial")
    spatial.add_argument("--checkpoint", type=Path, required=True)
    spatial.add_argument("--bank", type=Path, required=True)
    spatial.add_argument("--tasks", type=_integers, default=(7, 9))
    spatial.add_argument("--episodes", type=_integers, default=(0, 4, 6, 8))
    spatial.add_argument("--conditions", default="aligned,conflict")
    spatial.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    spatial.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit-spatial":
        bank, spatial_fit_report = fit_spatial_location_probe_bank(
            args.checkpoint,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            representation_layers=args.representation_layers,
            representation_position=args.representation_position,
            device=args.device,
            noise_seed=args.noise_seed,
            max_steps=args.max_steps,
            destination_rollout_task_ids=args.destination_rollout_tasks or (),
            destination_rollout_episode_indices=(
                args.destination_rollout_episodes or ()
            ),
            in_hand_sample_stride=args.in_hand_sample_stride,
            in_hand_min_steps_after_grasp=args.in_hand_min_steps_after_grasp,
            grasp_close_fraction=args.grasp_close_fraction,
            grasp_patience=args.grasp_patience,
            progress=lambda state: print(
                f"[location-monitor-spatial-fit] task={state.task_id} "
                f"episode={state.episode_index} condition={state.condition} "
                f"label={state.location_label}",
                file=sys.stderr,
                flush=True,
            ),
            trajectory_progress=lambda trajectory: print(
                f"[location-monitor-spatial-rollout] task={trajectory.task_id} "
                f"episode={trajectory.episode_index} success={trajectory.success} "
                f"steps={trajectory.steps}",
                file=sys.stderr,
                flush=True,
            ),
        )
        save_location_probe_bank(bank, args.bank_output)
        _write_report(args.output, spatial_fit_report.to_dict())
        return 0
    if args.command == "evaluate-spatial":
        conditions = tuple(
            item.strip() for item in args.conditions.split(",") if item.strip()
        )
        if not set(conditions) <= {"aligned", "conflict", "destination"}:
            raise ValueError("Spatial monitor condition is unknown")
        spatial_report = evaluate_location_probe_spatial_interventions(
            args.checkpoint,
            args.bank,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            conditions=cast(tuple[SpatialLocationTrainingCondition, ...], conditions),
            device=args.device,
            progress=lambda example: print(
                f"[location-monitor-spatial] task={example.task_id} "
                f"episode={example.episode_index} condition={example.condition} "
                f"actual={example.actual_label} predicted={example.predicted_label} "
                f"status={example.monitor_status}",
                file=sys.stderr,
                flush=True,
            ),
        )
        _write_report(args.output, spatial_report.to_dict())
        return 0
    if args.command == "evaluate":
        evaluation_report = evaluate_location_probe_initial_states(
            args.checkpoint,
            args.bank,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            device=args.device,
            progress=lambda example: print(
                f"[location-monitor-eval] task={example.task_id} "
                f"episode={example.episode_index} predicted={example.predicted_label} "
                f"aligned={example.aligned_status} conflict={example.conflict_status}",
                file=sys.stderr,
                flush=True,
            ),
        )
        _write_report(args.output, evaluation_report.to_dict())
        return 0
    bank, report = fit_location_probe_bank(
        args.checkpoint,
        task_ids=args.tasks,
        episode_indices=args.episodes,
        representation_layers=args.representation_layers,
        representation_position=args.representation_position,
        noise_seed=args.noise_seed,
        device=args.device,
        progress=lambda outcome: print(
            f"[location-monitor] task={outcome.task_id} episode={outcome.episode_index} "
            f"success={outcome.success} steps={outcome.steps}",
            file=sys.stderr,
            flush=True,
        ),
    )
    save_location_probe_bank(bank, args.bank_output)
    _write_report(args.output, report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
