"""Object-conditioned visual presence monitoring for absent-object conflicts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.location_monitor_runtime import capture_location_representations
from causal_vla.monitor_calibration import (
    LabeledMonitorExample,
    MonitorCalibration,
    calibrate_conflict_monitor,
)
from causal_vla.object_intervention import (
    ObjectPresenceCondition,
    apply_hidden_object_scene,
    object_conflict_task,
)
from causal_vla.probe_monitor_runtime import (
    ProbeDataset,
    fit_ridge_probe,
    leave_episode_out_logits,
    probe_logits,
)
from causal_vla.routing import ConflictAssessment, ConflictDetector, EvidenceDistribution
from causal_vla.smoke import prepare_libero_batch

_PRESENCE_LABELS = ("absent", "present")
_PROPOSITION = "claimed_object_presence"


def language_presence_evidence(claimed_label: str) -> EvidenceDistribution:
    """Encode the instruction's visually falsifiable object-presence claim."""

    if claimed_label not in _PRESENCE_LABELS:
        raise ValueError("Claimed presence is outside the monitor vocabulary")
    probabilities = tuple(0.98 if label == claimed_label else 0.02 for label in _PRESENCE_LABELS)
    return EvidenceDistribution("language", _PROPOSITION, _PRESENCE_LABELS, probabilities)


def presence_vision_evidence(logits: Tensor, *, temperature: float) -> EvidenceDistribution:
    """Convert binary probe scores into calibrated presence probabilities."""

    if logits.shape != (2,) or temperature <= 0:
        raise ValueError("Presence logits or temperature are invalid")
    probabilities = torch.softmax(logits.double() / temperature, dim=0)
    return EvidenceDistribution(
        "vision",
        _PROPOSITION,
        _PRESENCE_LABELS,
        tuple(float(value) for value in probabilities),
    )


@dataclass(frozen=True)
class PresenceProbeBank:
    """Tensor-only object-conditioned probe and frozen detector thresholds."""

    representation_layer: int
    representation_position: int
    weight: Tensor
    alpha: float
    temperature: float
    divergence_threshold: float
    confidence_floor: float
    training_task_id: int
    training_object_names: tuple[str, ...]
    training_episode_indices: tuple[int, ...]
    calibration_episode_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[1] != 2:
            raise ValueError("Presence probe weight must produce two scores")
        if self.weight.shape[0] < 2 or self.alpha <= 0 or self.temperature <= 0:
            raise ValueError("Presence probe hyperparameters are invalid")
        if not self.training_object_names or len(set(self.training_object_names)) != len(
            self.training_object_names
        ):
            raise ValueError("Training object names must be nonempty and unique")
        if not 0 <= self.divergence_threshold <= 1:
            raise ValueError("Presence divergence threshold must lie in [0, 1]")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("Presence confidence floor must lie in [0, 1]")
        if set(self.training_episode_indices) & set(self.calibration_episode_indices):
            raise ValueError("Presence fit and calibration episodes must be disjoint")
        if set(self.training_episode_indices + self.calibration_episode_indices) & set(
            range(10, 15)
        ):
            raise ValueError("Historical held-out episodes 10--14 are sealed")


def save_presence_probe_bank(bank: PresenceProbeBank, path: str | Path) -> None:
    """Save a weights-only compatible presence monitor artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema_version": 1, **asdict(bank)}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_presence_probe_bank(path: str | Path) -> PresenceProbeBank:
    """Load and validate a tensor-only presence monitor."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported presence probe-bank schema")
    return PresenceProbeBank(
        representation_layer=int(payload["representation_layer"]),
        representation_position=int(payload["representation_position"]),
        weight=cast(Tensor, payload["weight"]),
        alpha=float(payload["alpha"]),
        temperature=float(payload["temperature"]),
        divergence_threshold=float(payload["divergence_threshold"]),
        confidence_floor=float(payload["confidence_floor"]),
        training_task_id=int(payload["training_task_id"]),
        training_object_names=tuple(str(value) for value in payload["training_object_names"]),
        training_episode_indices=tuple(int(value) for value in payload["training_episode_indices"]),
        calibration_episode_indices=tuple(
            int(value) for value in payload.get("calibration_episode_indices", ())
        ),
    )


def _presence_examples(
    dataset: ProbeDataset, logits: Tensor, *, temperature: float
) -> tuple[LabeledMonitorExample, ...]:
    """Build only the instruction-relevant claim that the named object is present."""

    examples: list[LabeledMonitorExample] = []
    for row in range(dataset.features.shape[0]):
        true_label = dataset.labels[int(dataset.label_indices[row])]
        examples.append(
            LabeledMonitorExample(
                task_id=int(dataset.task_ids[row]),
                episode_index=int(dataset.episode_indices[row]),
                conflict_type="absent_claimed_object",
                expected_status="aligned" if true_label == "present" else "conflict",
                true_visual_label=true_label,
                vision=presence_vision_evidence(logits[row], temperature=temperature),
                language=language_presence_evidence("present"),
            )
        )
    return tuple(examples)


@dataclass(frozen=True)
class PresenceSelectionRow:
    representation_layer: int
    alpha: float
    temperature: float
    binary_accuracy: float
    calibration: MonitorCalibration


@dataclass(frozen=True)
class PresenceTrainingState:
    task_id: int
    episode_index: int
    object_name: str
    condition: ObjectPresenceCondition
    hidden_object: str


@dataclass(frozen=True)
class PresenceProbeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    object_names: tuple[str, ...]
    episode_indices: tuple[int, ...]
    representation_layers: tuple[int, ...]
    representation_position: int
    states: tuple[PresenceTrainingState, ...]
    selected_layer: int
    selected_alpha: float
    selected_temperature: float
    calibration: MonitorCalibration
    selection_rows: tuple[PresenceSelectionRow, ...]

    def to_dict(self) -> dict[str, object]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"states", "calibration", "selection_rows"}
        }
        payload["states"] = [asdict(state) for state in self.states]
        payload["calibration"] = self.calibration.to_dict()
        payload["selection_rows"] = [
            {
                "representation_layer": row.representation_layer,
                "alpha": row.alpha,
                "temperature": row.temperature,
                "binary_accuracy": row.binary_accuracy,
                "selected_operating_point": asdict(row.calibration.selected),
            }
            for row in self.selection_rows
        ]
        return payload


def select_presence_probe(
    datasets: Mapping[int, ProbeDataset],
    *,
    representation_position: int,
    task_id: int,
    object_names: tuple[str, ...],
    alpha_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    temperature_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> tuple[PresenceProbeBank, tuple[PresenceSelectionRow, ...], MonitorCalibration]:
    """Select representation and thresholds using leave-episode-out predictions."""

    if not datasets or not alpha_candidates or not temperature_candidates:
        raise ValueError("Presence-probe selection candidates must be nonempty")
    rows: list[PresenceSelectionRow] = []
    for layer, dataset in datasets.items():
        for alpha in alpha_candidates:
            logits = leave_episode_out_logits(dataset, alpha=alpha)
            accuracy = float((logits.argmax(dim=1) == dataset.label_indices).double().mean())
            for temperature in temperature_candidates:
                calibration = calibrate_conflict_monitor(
                    _presence_examples(dataset, logits, temperature=temperature),
                    divergence_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
                    confidence_candidates=(
                        0.50,
                        0.55,
                        0.60,
                        0.65,
                        0.70,
                        0.75,
                        0.80,
                        0.85,
                        0.90,
                        0.95,
                    ),
                    max_aligned_false_trigger_rate=0.0,
                )
                rows.append(PresenceSelectionRow(layer, alpha, temperature, accuracy, calibration))
    selected = max(
        rows,
        key=lambda row: (
            -row.calibration.selected.aligned_false_trigger_rate,
            row.calibration.selected.conflict_recall,
            row.binary_accuracy,
            -row.calibration.selected.aligned_abstention_rate,
            -row.alpha,
            -row.temperature,
            -row.representation_layer,
        ),
    )
    dataset = datasets[selected.representation_layer]
    bank = PresenceProbeBank(
        representation_layer=selected.representation_layer,
        representation_position=representation_position,
        weight=fit_ridge_probe(
            dataset.features, dataset.label_indices, classes=2, alpha=selected.alpha
        ),
        alpha=selected.alpha,
        temperature=selected.temperature,
        divergence_threshold=selected.calibration.selected.divergence_threshold,
        confidence_floor=selected.calibration.selected.confidence_floor,
        training_task_id=task_id,
        training_object_names=object_names,
        training_episode_indices=tuple(sorted(set(dataset.episode_indices.tolist()))),
    )
    return bank, tuple(rows), selected.calibration


def _validate_development_episodes(episode_indices: tuple[int, ...]) -> None:
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episodes must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")


def _validate_object_names(task_id: int, object_names: tuple[str, ...]) -> None:
    if not object_names or len(set(object_names)) != len(object_names):
        raise ValueError("Object names must be nonempty and unique")
    task = object_conflict_task(task_id)
    expected = {task.target_object, task.hidden_object}
    if set(object_names) != expected:
        raise ValueError("Presence fitting must counterbalance both target and conflicting objects")


def _counterbalanced_hidden_object(
    object_names: tuple[str, ...],
    *,
    queried_object: str,
    condition: ObjectPresenceCondition,
) -> str:
    """Hide one object while making the label depend on the queried identity."""

    if len(object_names) != 2 or queried_object not in object_names:
        raise ValueError("Counterbalanced presence monitoring requires two known objects")
    if condition == "absent":
        return queried_object
    return next(name for name in object_names if name != queried_object)


def fit_presence_probe_bank(
    checkpoint: str | Path,
    *,
    task_id: int = 4,
    object_names: tuple[str, ...] = ("ketchup", "milk"),
    episode_indices: tuple[int, ...] = (1, 2, 3, 5, 7, 9),
    representation_layers: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 31),
    representation_position: int = -2,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[PresenceTrainingState], None] | None = None,
) -> tuple[PresenceProbeBank, PresenceProbeFitReport]:
    """Fit a query-conditioned probe from globally balanced interventions.

    Every observation has exactly one object hidden.  Thus the same milk-missing
    scene is absent for a milk query and present for a ketchup query, preventing
    the probe from using a global missing-object cue.
    """

    _validate_development_episodes(episode_indices)
    _validate_object_names(task_id, object_names)
    if not representation_layers or len(set(representation_layers)) != len(representation_layers):
        raise ValueError("Representation layers must be nonempty and unique")
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    feature_lists: dict[int, list[Tensor]] = {layer: [] for layer in representation_layers}
    label_indices: list[int] = []
    row_tasks: list[int] = []
    row_episodes: list[int] = []
    states: list[PresenceTrainingState] = []
    task = object_conflict_task(task_id)
    for episode_index in episode_indices:
        for object_name in object_names:
            for condition in cast(tuple[ObjectPresenceCondition, ...], ("present", "absent")):
                environment, native_prompt, _ = adapter.create_env(
                    task_id,
                    suite="libero_object",
                    resolution=resolution,
                    episode_index=episode_index,
                )
                try:
                    if native_prompt.casefold() != task.native_prompt.casefold():
                        raise ValueError("Native instruction differs from object conflict task")
                    environment.reset(seed=simulator_seed + episode_index)
                    hidden_object = _counterbalanced_hidden_object(
                        object_names,
                        queried_object=object_name,
                        condition=condition,
                    )
                    observation, _ = apply_hidden_object_scene(
                        environment, object_name=hidden_object
                    )
                    batch = prepare_libero_batch(
                        adapter,
                        observation,
                        f"Is the {object_name} visible right now?",
                    )
                    captures = capture_location_representations(
                        policy,
                        batch,
                        representation_layers=representation_layers,
                        representation_position=representation_position,
                    )
                finally:
                    environment.close()
                for layer, feature in captures.items():
                    feature_lists[layer].append(feature)
                label_indices.append(_PRESENCE_LABELS.index(condition))
                row_tasks.append(task_id)
                row_episodes.append(episode_index)
                state = PresenceTrainingState(
                    task_id, episode_index, object_name, condition, hidden_object
                )
                states.append(state)
                if progress is not None:
                    progress(state)
    datasets = {
        layer: ProbeDataset(
            features=torch.cat(features),
            label_indices=torch.tensor(label_indices, dtype=torch.int64),
            task_ids=torch.tensor(row_tasks, dtype=torch.int64),
            episode_indices=torch.tensor(row_episodes, dtype=torch.int64),
            labels=_PRESENCE_LABELS,
        )
        for layer, features in feature_lists.items()
    }
    bank, selection_rows, calibration = select_presence_probe(
        datasets,
        representation_position=representation_position,
        task_id=task_id,
        object_names=object_names,
    )
    report = PresenceProbeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite="libero_object",
        task_id=task_id,
        object_names=object_names,
        episode_indices=episode_indices,
        representation_layers=representation_layers,
        representation_position=representation_position,
        states=tuple(states),
        selected_layer=bank.representation_layer,
        selected_alpha=bank.alpha,
        selected_temperature=bank.temperature,
        calibration=calibration,
        selection_rows=selection_rows,
    )
    return bank, report


def predict_presence_evidence(
    policy: Any,
    adapter: Any,
    observation: dict[str, Any],
    *,
    object_name: str,
    bank: PresenceProbeBank,
) -> EvidenceDistribution:
    """Predict whether a named object is visible using an object-conditioned query."""

    batch = prepare_libero_batch(adapter, observation, f"Is the {object_name} visible right now?")
    representation = capture_location_representations(
        policy,
        batch,
        representation_layers=(bank.representation_layer,),
        representation_position=bank.representation_position,
    )[bank.representation_layer]
    return presence_vision_evidence(
        probe_logits(representation, bank.weight)[0], temperature=bank.temperature
    )


def assess_presence_claim(
    policy: Any,
    adapter: Any,
    observation: dict[str, Any],
    *,
    object_name: str,
    bank: PresenceProbeBank,
) -> ConflictAssessment:
    """Assess the instruction's implicit claim that its named object is present."""

    vision = predict_presence_evidence(
        policy, adapter, observation, object_name=object_name, bank=bank
    )
    return ConflictDetector(
        divergence_threshold=bank.divergence_threshold,
        confidence_floor=bank.confidence_floor,
    ).assess(vision, language_presence_evidence("present"))


@dataclass(frozen=True)
class PresenceMonitorExample:
    episode_index: int
    object_name: str
    condition: ObjectPresenceCondition
    hidden_object: str
    predicted_label: str
    predicted_confidence: float
    monitor_status: str
    target_correct: bool


@dataclass(frozen=True)
class PresenceMonitorEvaluationReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    suite: str
    task_id: int
    object_names: tuple[str, ...]
    episode_indices: tuple[int, ...]
    examples: tuple[PresenceMonitorExample, ...]
    binary_accuracy: float
    aligned_false_trigger_rate: float
    conflict_recall: float

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["examples"] = [asdict(example) for example in self.examples]
        return payload


@dataclass(frozen=True)
class PresenceThresholdCalibrationReport:
    """Auditable development-only threshold calibration from frozen predictions."""

    schema_version: int
    input_bank_path: str
    input_evaluation_path: str
    task_id: int
    object_names: tuple[str, ...]
    training_episode_indices: tuple[int, ...]
    calibration_episode_indices: tuple[int, ...]
    calibration: MonitorCalibration

    def to_dict(self) -> dict[str, object]:
        payload = {key: value for key, value in asdict(self).items() if key != "calibration"}
        payload["calibration"] = self.calibration.to_dict()
        return payload


def calibrate_presence_probe_bank(
    bank_path: str | Path,
    evaluation_path: str | Path,
) -> tuple[PresenceProbeBank, PresenceThresholdCalibrationReport]:
    """Freeze thresholds from a disjoint development prediction report."""

    bank = load_presence_probe_bank(bank_path)
    evaluation_source = Path(evaluation_path).resolve()
    payload = json.loads(evaluation_source.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported presence-monitor evaluation schema")
    task_id = int(payload["task_id"])
    object_names = tuple(str(value) for value in payload["object_names"])
    episode_indices = tuple(int(value) for value in payload["episode_indices"])
    _validate_development_episodes(episode_indices)
    if task_id != bank.training_task_id or set(object_names) != set(bank.training_object_names):
        raise ValueError("Calibration report does not match monitor provenance")
    if set(episode_indices) & set(bank.training_episode_indices):
        raise ValueError("Presence calibration episodes overlap probe-fit episodes")
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("Presence calibration report has no examples")
    examples: list[LabeledMonitorExample] = []
    for raw in raw_examples:
        if not isinstance(raw, dict):
            raise ValueError("Presence calibration example is malformed")
        condition = str(raw["condition"])
        predicted_label = str(raw["predicted_label"])
        confidence = float(raw["predicted_confidence"])
        if condition not in _PRESENCE_LABELS or predicted_label not in _PRESENCE_LABELS:
            raise ValueError("Presence calibration labels are invalid")
        if not 0.5 <= confidence <= 1.0:
            raise ValueError("Binary presence confidence must lie in [0.5, 1]")
        probabilities = tuple(
            confidence if label == predicted_label else 1.0 - confidence
            for label in _PRESENCE_LABELS
        )
        examples.append(
            LabeledMonitorExample(
                task_id=task_id,
                episode_index=int(raw["episode_index"]),
                conflict_type="absent_claimed_object",
                expected_status="aligned" if condition == "present" else "conflict",
                true_visual_label=condition,
                vision=EvidenceDistribution(
                    "vision", _PROPOSITION, _PRESENCE_LABELS, probabilities
                ),
                language=language_presence_evidence("present"),
            )
        )
    calibration = calibrate_conflict_monitor(
        tuple(examples),
        divergence_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
        confidence_candidates=(
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ),
        max_aligned_false_trigger_rate=0.0,
    )
    calibrated = replace(
        bank,
        divergence_threshold=calibration.selected.divergence_threshold,
        confidence_floor=calibration.selected.confidence_floor,
        calibration_episode_indices=episode_indices,
    )
    report = PresenceThresholdCalibrationReport(
        schema_version=1,
        input_bank_path=str(Path(bank_path).resolve()),
        input_evaluation_path=str(evaluation_source),
        task_id=task_id,
        object_names=object_names,
        training_episode_indices=bank.training_episode_indices,
        calibration_episode_indices=episode_indices,
        calibration=calibration,
    )
    return calibrated, report


def evaluate_presence_probe_bank(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    task_id: int = 4,
    object_names: tuple[str, ...] = ("ketchup", "milk"),
    episode_indices: tuple[int, ...] = (0, 4, 6, 8),
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[PresenceMonitorExample], None] | None = None,
) -> PresenceMonitorEvaluationReport:
    """Evaluate unseen scenes while keeping one object hidden in every example."""

    _validate_development_episodes(episode_indices)
    _validate_object_names(task_id, object_names)
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    bank = load_presence_probe_bank(bank_path)
    if bank.training_task_id != task_id or set(bank.training_object_names) != set(object_names):
        raise ValueError("Presence monitor provenance does not match evaluation task")
    task = object_conflict_task(task_id)
    examples: list[PresenceMonitorExample] = []
    for episode_index in episode_indices:
        for object_name in object_names:
            for condition in cast(tuple[ObjectPresenceCondition, ...], ("present", "absent")):
                environment, native_prompt, _ = adapter.create_env(
                    task_id,
                    suite="libero_object",
                    resolution=resolution,
                    episode_index=episode_index,
                )
                try:
                    if native_prompt.casefold() != task.native_prompt.casefold():
                        raise ValueError("Native instruction differs from object conflict task")
                    environment.reset(seed=simulator_seed + episode_index)
                    hidden_object = _counterbalanced_hidden_object(
                        object_names,
                        queried_object=object_name,
                        condition=condition,
                    )
                    observation, _ = apply_hidden_object_scene(
                        environment, object_name=hidden_object
                    )
                    assessment = assess_presence_claim(
                        policy,
                        adapter,
                        observation,
                        object_name=object_name,
                        bank=bank,
                    )
                finally:
                    environment.close()
                example = PresenceMonitorExample(
                    episode_index=episode_index,
                    object_name=object_name,
                    condition=condition,
                    hidden_object=hidden_object,
                    predicted_label=assessment.vision_label,
                    predicted_confidence=assessment.vision_confidence,
                    monitor_status=assessment.status,
                    target_correct=assessment.vision_label == condition,
                )
                examples.append(example)
                if progress is not None:
                    progress(example)
    aligned = tuple(example for example in examples if example.condition == "present")
    conflicts = tuple(example for example in examples if example.condition == "absent")
    return PresenceMonitorEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        suite="libero_object",
        task_id=task_id,
        object_names=object_names,
        episode_indices=episode_indices,
        examples=tuple(examples),
        binary_accuracy=sum(example.target_correct for example in examples) / len(examples),
        aligned_false_trigger_rate=(
            sum(example.monitor_status == "conflict" for example in aligned) / len(aligned)
        ),
        conflict_recall=(
            sum(
                example.monitor_status == "conflict" and example.target_correct
                for example in conflicts
            )
            / len(conflicts)
        ),
    )


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def _strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated strings")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.object_monitor_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--task-id", type=int, default=4)
    fit.add_argument("--objects", type=_strings, default=("ketchup", "milk"))
    fit.add_argument("--episodes", type=_integers, default=(1, 2, 3, 5, 7, 9))
    fit.add_argument("--representation-layers", type=_integers, default=(8, 12, 16, 20, 24, 28, 31))
    fit.add_argument("--representation-position", type=int, default=-2)
    fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    fit.add_argument("--bank-output", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--task-id", type=int, default=4)
    evaluate.add_argument("--objects", type=_strings, default=("ketchup", "milk"))
    evaluate.add_argument("--episodes", type=_integers, default=(0, 4, 6, 8))
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    evaluate.add_argument("--output", type=Path, required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--bank", type=Path, required=True)
    calibrate.add_argument("--evaluation", type=Path, required=True)
    calibrate.add_argument("--bank-output", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    return parser


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        bank, fit_report = fit_presence_probe_bank(
            args.checkpoint,
            task_id=args.task_id,
            object_names=args.objects,
            episode_indices=args.episodes,
            representation_layers=args.representation_layers,
            representation_position=args.representation_position,
            device=args.device,
            progress=lambda state: print(
                f"[presence-fit] episode={state.episode_index} "
                f"object={state.object_name} condition={state.condition}",
                file=sys.stderr,
                flush=True,
            ),
        )
        save_presence_probe_bank(bank, args.bank_output)
        _write_report(args.output, fit_report.to_dict())
        return 0
    if args.command == "calibrate":
        bank, calibration_report = calibrate_presence_probe_bank(args.bank, args.evaluation)
        save_presence_probe_bank(bank, args.bank_output)
        _write_report(args.output, calibration_report.to_dict())
        return 0
    evaluation_report = evaluate_presence_probe_bank(
        args.checkpoint,
        args.bank,
        task_id=args.task_id,
        object_names=args.objects,
        episode_indices=args.episodes,
        device=args.device,
        progress=lambda example: print(
            f"[presence-eval] episode={example.episode_index} "
            f"object={example.object_name} condition={example.condition} "
            f"prediction={example.predicted_label} status={example.monitor_status}",
            file=sys.stderr,
            flush=True,
        ),
    )
    _write_report(args.output, evaluation_report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
