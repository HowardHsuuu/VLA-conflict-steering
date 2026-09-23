"""Linear visual-state monitor on frozen, location-neutral VLM representations."""

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
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.interventions import InputActivationCapture
from causal_vla.monitor_calibration import (
    LabeledMonitorExample,
    MonitorCalibration,
    calibrate_conflict_monitor,
    evaluate_monitor_operating_point,
)
from causal_vla.monitor_scan_runtime import SupportConflictSpec, support_conflict_specs
from causal_vla.residual_runtime import vlm_layers
from causal_vla.routing import EvidenceDistribution
from causal_vla.smoke import prepare_libero_batch

_NEUTRAL_PROMPT = "pick up the black bowl and place it on the plate"
_PROPOSITION = "initial_black_bowl_support"
RepresentationPool = Literal["token", "image_mean", "camera_means", "image_tokens"]


def monitor_representation(
    policy: Any,
    batch: dict[str, Any],
    *,
    representation_layer: int,
    representation_position: int,
    representation_pool: RepresentationPool,
) -> Tensor:
    """Capture one location-neutral VLM residual summary."""

    layers, _ = vlm_layers(policy)
    if not 0 <= representation_layer < len(layers):
        raise ValueError("Representation layer is outside the VLM decoder")
    capture = InputActivationCapture(dtype=torch.float32)
    handle = layers[representation_layer].input_layernorm.register_forward_pre_hook(capture)
    try:
        snapshot = prefix_snapshot(policy, batch)
    finally:
        handle.remove()
    if len(capture.records) != 1:
        raise RuntimeError("Expected one VLM representation capture")
    residual = capture.records[0]
    if representation_pool == "token":
        if not -residual.shape[1] <= representation_position < residual.shape[1]:
            raise ValueError("Representation position is outside the prefix")
        return residual[..., representation_position, :].reshape(1, -1)
    image_total = sum(snapshot.image_lengths)
    if image_total <= 0:
        raise ValueError("Prefix contains no image tokens")
    if representation_pool == "image_mean":
        return residual[:, :image_total].mean(dim=1)
    if representation_pool == "image_tokens":
        return residual[:, :image_total].reshape(1, -1)
    if representation_pool == "camera_means":
        offset = 0
        means: list[Tensor] = []
        for length in snapshot.image_lengths:
            means.append(residual[:, offset : offset + length].mean(dim=1))
            offset += length
        return torch.cat(means, dim=1)
    raise ValueError(f"Unknown representation pool {representation_pool!r}")


@dataclass(frozen=True)
class ProbeDataset:
    """Frozen initial-state VLM features and categorical source labels."""

    features: Tensor
    label_indices: Tensor
    task_ids: Tensor
    episode_indices: Tensor
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        if self.features.ndim != 2 or count == 0:
            raise ValueError("Probe features must be a nonempty matrix")
        if any(
            value.shape != (count,)
            for value in (
                self.label_indices,
                self.task_ids,
                self.episode_indices,
            )
        ):
            raise ValueError("Probe metadata must align with feature rows")
        if not self.labels or int(self.label_indices.min()) < 0:
            raise ValueError("Probe labels are empty or invalid")
        if int(self.label_indices.max()) >= len(self.labels):
            raise ValueError("Probe label index exceeds label vocabulary")


@dataclass(frozen=True)
class VisualProbeBank:
    """Tensor-only linear monitor with frozen calibration thresholds."""

    labels: tuple[str, ...]
    representation_layer: int
    representation_position: int
    representation_pool: RepresentationPool
    neutral_prompt: str
    weight: Tensor
    alpha: float
    temperature: float
    divergence_threshold: float
    confidence_floor: float
    training_task_ids: tuple[int, ...]
    training_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[1] != len(self.labels):
            raise ValueError("Probe weight must map augmented features to labels")
        if self.weight.shape[0] < 2:
            raise ValueError("Probe weight must include features and an intercept")
        if self.temperature <= 0 or self.alpha <= 0:
            raise ValueError("Probe alpha and temperature must be positive")


@dataclass(frozen=True)
class PairwiseProbe:
    """One calibrated linear monitor for a causally supported label pair."""

    labels: tuple[str, str]
    weight: Tensor
    alpha: float
    temperature: float
    divergence_threshold: float
    confidence_floor: float

    def __post_init__(self) -> None:
        if self.labels[0] == self.labels[1] or self.weight.shape[1] != 2:
            raise ValueError("Pairwise probe requires two distinct labels and two scores")
        if self.alpha <= 0 or self.temperature <= 0:
            raise ValueError("Pairwise probe alpha and temperature must be positive")


@dataclass(frozen=True)
class PairwiseVisualProbeBank:
    """Collection of calibrated transition-specific visual probes."""

    representation_layer: int
    representation_position: int
    representation_pool: RepresentationPool
    neutral_prompt: str
    probes: dict[str, PairwiseProbe]
    training_task_ids: tuple[int, ...]
    training_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("Pairwise monitor bank must contain at least one probe")
        for key, probe in self.probes.items():
            if key != pair_key(probe.labels):
                raise ValueError("Pairwise probe key does not match its labels")


def pair_key(labels: tuple[str, str]) -> str:
    """Return an order-invariant key for one supported label transition."""

    if labels[0] == labels[1]:
        raise ValueError("A monitor pair must contain distinct labels")
    return "|".join(sorted(labels))


def save_pairwise_visual_probe_bank(bank: PairwiseVisualProbeBank, path: str | Path) -> None:
    """Save pairwise probes as primitive values and tensors only."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "representation_layer": bank.representation_layer,
        "representation_position": bank.representation_position,
        "representation_pool": bank.representation_pool,
        "neutral_prompt": bank.neutral_prompt,
        "training_task_ids": bank.training_task_ids,
        "training_episode_indices": bank.training_episode_indices,
        "probes": {
            key: {
                "labels": probe.labels,
                "weight": probe.weight,
                "alpha": probe.alpha,
                "temperature": probe.temperature,
                "divergence_threshold": probe.divergence_threshold,
                "confidence_floor": probe.confidence_floor,
            }
            for key, probe in bank.probes.items()
        },
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_pairwise_visual_probe_bank(path: str | Path) -> PairwiseVisualProbeBank:
    """Load and validate a weights-only compatible pairwise monitor bank."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported pairwise visual probe-bank schema")
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, dict):
        raise ValueError("Pairwise monitor bank is missing probes")
    probes: dict[str, PairwiseProbe] = {}
    for key, raw in raw_probes.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            raise ValueError("Pairwise monitor probe entry is malformed")
        labels = tuple(str(value) for value in raw["labels"])
        if len(labels) != 2:
            raise ValueError("Pairwise monitor labels are malformed")
        probes[key] = PairwiseProbe(
            labels=labels,
            weight=cast(Tensor, raw["weight"]),
            alpha=float(raw["alpha"]),
            temperature=float(raw["temperature"]),
            divergence_threshold=float(raw["divergence_threshold"]),
            confidence_floor=float(raw["confidence_floor"]),
        )
    return PairwiseVisualProbeBank(
        representation_layer=int(payload["representation_layer"]),
        representation_position=int(payload["representation_position"]),
        representation_pool=cast(RepresentationPool, payload.get("representation_pool", "token")),
        neutral_prompt=str(payload["neutral_prompt"]),
        probes=probes,
        training_task_ids=tuple(int(value) for value in payload["training_task_ids"]),
        training_episode_indices=tuple(int(value) for value in payload["training_episode_indices"]),
    )


def save_visual_probe_bank(bank: VisualProbeBank, path: str | Path) -> None:
    """Save a weights-only compatible visual monitor artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "labels": bank.labels,
        "representation_layer": bank.representation_layer,
        "representation_position": bank.representation_position,
        "representation_pool": bank.representation_pool,
        "neutral_prompt": bank.neutral_prompt,
        "weight": bank.weight,
        "alpha": bank.alpha,
        "temperature": bank.temperature,
        "divergence_threshold": bank.divergence_threshold,
        "confidence_floor": bank.confidence_floor,
        "training_task_ids": bank.training_task_ids,
        "training_episode_indices": bank.training_episode_indices,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_visual_probe_bank(path: str | Path) -> VisualProbeBank:
    """Load and validate a tensor-only visual monitor artifact."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported visual probe-bank schema")
    return VisualProbeBank(
        labels=tuple(str(value) for value in payload["labels"]),
        representation_layer=int(payload["representation_layer"]),
        representation_position=int(payload["representation_position"]),
        representation_pool=cast(RepresentationPool, payload.get("representation_pool", "token")),
        neutral_prompt=str(payload["neutral_prompt"]),
        weight=cast(Tensor, payload["weight"]),
        alpha=float(payload["alpha"]),
        temperature=float(payload["temperature"]),
        divergence_threshold=float(payload["divergence_threshold"]),
        confidence_floor=float(payload["confidence_floor"]),
        training_task_ids=tuple(int(value) for value in payload["training_task_ids"]),
        training_episode_indices=tuple(int(value) for value in payload["training_episode_indices"]),
    )


def _normalized_augmented(features: Tensor) -> Tensor:
    normalized = torch.nn.functional.normalize(features.double(), dim=1)
    return torch.cat(
        (normalized, torch.ones((normalized.shape[0], 1), dtype=normalized.dtype)), dim=1
    )


def fit_ridge_probe(
    features: Tensor, label_indices: Tensor, *, classes: int, alpha: float
) -> Tensor:
    """Fit one-vs-rest ridge scores through the sample-space dual."""

    if alpha <= 0 or classes <= 1:
        raise ValueError("Ridge alpha must be positive and classes must exceed one")
    x = _normalized_augmented(features)
    targets = torch.nn.functional.one_hot(label_indices.long(), classes).double()
    gram = x @ x.transpose(0, 1)
    regularized = gram + alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
    return cast(Tensor, x.transpose(0, 1) @ torch.linalg.solve(regularized, targets))


def probe_logits(features: Tensor, weight: Tensor) -> Tensor:
    """Apply a fitted ridge probe to frozen representation rows."""

    x = _normalized_augmented(features)
    if x.shape[1] != weight.shape[0]:
        raise ValueError("Probe feature dimension does not match its weight")
    return x @ weight.double()


def leave_episode_out_logits(dataset: ProbeDataset, *, alpha: float) -> Tensor:
    """Predict every row using a probe that excludes its episode index."""

    result = torch.empty((dataset.features.shape[0], len(dataset.labels)), dtype=torch.float64)
    for episode_index in torch.unique(dataset.episode_indices).tolist():
        held_out = dataset.episode_indices == int(episode_index)
        training = ~held_out
        weight = fit_ridge_probe(
            dataset.features[training],
            dataset.label_indices[training],
            classes=len(dataset.labels),
            alpha=alpha,
        )
        result[held_out] = probe_logits(dataset.features[held_out], weight)
    return result


def _pairwise_evidence(
    logits: Tensor,
    *,
    labels: tuple[str, ...],
    pair: tuple[str, str],
    temperature: float,
) -> EvidenceDistribution:
    indices = torch.tensor([labels.index(label) for label in pair], dtype=torch.int64)
    probabilities = torch.softmax(logits[indices] / temperature, dim=0)
    return EvidenceDistribution(
        "vision", _PROPOSITION, pair, tuple(float(value) for value in probabilities)
    )


def _language_evidence(pair: tuple[str, str], instruction_label: str) -> EvidenceDistribution:
    probabilities = tuple(0.98 if label == instruction_label else 0.02 for label in pair)
    return EvidenceDistribution("language", _PROPOSITION, pair, probabilities)


def _calibration_examples(
    dataset: ProbeDataset,
    logits: Tensor,
    specs: Mapping[int, SupportConflictSpec],
    *,
    temperature: float,
) -> tuple[LabeledMonitorExample, ...]:
    examples: list[LabeledMonitorExample] = []
    for row in range(dataset.features.shape[0]):
        task_id = int(dataset.task_ids[row])
        episode_index = int(dataset.episode_indices[row])
        spec = specs[task_id]
        pair = (spec.true_label, spec.distractor_label)
        vision = _pairwise_evidence(
            logits[row], labels=dataset.labels, pair=pair, temperature=temperature
        )
        for expected, instruction_label in (
            ("aligned", spec.true_label),
            ("conflict", spec.distractor_label),
        ):
            examples.append(
                LabeledMonitorExample(
                    task_id=task_id,
                    episode_index=episode_index,
                    conflict_type=spec.conflict_type,
                    expected_status=cast(Any, expected),
                    true_visual_label=spec.true_label,
                    vision=vision,
                    language=_language_evidence(pair, instruction_label),
                )
            )
    return tuple(examples)


@dataclass(frozen=True)
class ProbeSelectionRow:
    alpha: float
    temperature: float
    multiclass_accuracy: float
    calibration: MonitorCalibration


@dataclass(frozen=True)
class ProbeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    neutral_prompt: str
    representation_layer: int
    representation_position: int
    representation_pool: RepresentationPool
    training_task_ids: tuple[int, ...]
    training_episode_indices: tuple[int, ...]
    labels: tuple[str, ...]
    samples: int
    selection_rows: tuple[ProbeSelectionRow, ...]
    selected_alpha: float
    selected_temperature: float
    calibration: MonitorCalibration

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"selection_rows", "calibration"}
            },
            "selection_rows": [
                {
                    "alpha": row.alpha,
                    "temperature": row.temperature,
                    "multiclass_accuracy": row.multiclass_accuracy,
                    "selected_operating_point": asdict(row.calibration.selected),
                }
                for row in self.selection_rows
            ],
            "calibration": self.calibration.to_dict(),
        }


def select_probe(
    dataset: ProbeDataset,
    specs: Mapping[int, SupportConflictSpec],
    *,
    alpha_candidates: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    temperature_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> tuple[VisualProbeBank, tuple[ProbeSelectionRow, ...], MonitorCalibration]:
    """Jointly select ridge and detector parameters using leave-episode-out evidence."""

    if not alpha_candidates or not temperature_candidates:
        raise ValueError("Probe selection candidates must be nonempty")
    rows: list[ProbeSelectionRow] = []
    for alpha in alpha_candidates:
        logits = leave_episode_out_logits(dataset, alpha=alpha)
        accuracy = float((logits.argmax(dim=1) == dataset.label_indices).double().mean())
        for temperature in temperature_candidates:
            examples = _calibration_examples(dataset, logits, specs, temperature=temperature)
            calibration = calibrate_conflict_monitor(
                examples,
                divergence_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
                confidence_candidates=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
                max_aligned_false_trigger_rate=0.0,
            )
            rows.append(ProbeSelectionRow(alpha, temperature, accuracy, calibration))
    selected = max(
        rows,
        key=lambda row: (
            -row.calibration.selected.aligned_false_trigger_rate,
            row.calibration.selected.conflict_recall,
            row.multiclass_accuracy,
            -row.calibration.selected.aligned_abstention_rate,
            -row.alpha,
            -row.temperature,
        ),
    )
    weight = fit_ridge_probe(
        dataset.features,
        dataset.label_indices,
        classes=len(dataset.labels),
        alpha=selected.alpha,
    )
    bank = VisualProbeBank(
        labels=dataset.labels,
        representation_layer=-1,
        representation_position=-1,
        representation_pool="token",
        neutral_prompt=_NEUTRAL_PROMPT,
        weight=weight,
        alpha=selected.alpha,
        temperature=selected.temperature,
        divergence_threshold=selected.calibration.selected.divergence_threshold,
        confidence_floor=selected.calibration.selected.confidence_floor,
        training_task_ids=tuple(sorted(set(dataset.task_ids.tolist()))),
        training_episode_indices=tuple(sorted(set(dataset.episode_indices.tolist()))),
    )
    return bank, tuple(rows), selected.calibration


def _pair_dataset(dataset: ProbeDataset, labels: tuple[str, str]) -> ProbeDataset:
    source_indices = tuple(dataset.labels.index(label) for label in labels)
    mask = (dataset.label_indices == source_indices[0]) | (
        dataset.label_indices == source_indices[1]
    )
    remapped = torch.where(
        dataset.label_indices[mask] == source_indices[0],
        torch.tensor(0, dtype=torch.int64),
        torch.tensor(1, dtype=torch.int64),
    )
    return ProbeDataset(
        features=dataset.features[mask],
        label_indices=remapped,
        task_ids=dataset.task_ids[mask],
        episode_indices=dataset.episode_indices[mask],
        labels=labels,
    )


def _pair_probe_examples(
    dataset: ProbeDataset,
    logits: Tensor,
    specs: Mapping[int, SupportConflictSpec],
    *,
    temperature: float,
) -> tuple[LabeledMonitorExample, ...]:
    examples: list[LabeledMonitorExample] = []
    expected_pair = pair_key(cast(tuple[str, str], dataset.labels))
    for row in range(dataset.features.shape[0]):
        task_id = int(dataset.task_ids[row])
        spec = specs[task_id]
        spec_pair = (spec.true_label, spec.distractor_label)
        if pair_key(spec_pair) != expected_pair:
            continue
        vision = _pairwise_evidence(
            logits[row],
            labels=dataset.labels,
            pair=spec_pair,
            temperature=temperature,
        )
        for expected, instruction_label in (
            ("aligned", spec.true_label),
            ("conflict", spec.distractor_label),
        ):
            examples.append(
                LabeledMonitorExample(
                    task_id=task_id,
                    episode_index=int(dataset.episode_indices[row]),
                    conflict_type=spec.conflict_type,
                    expected_status=cast(Any, expected),
                    true_visual_label=spec.true_label,
                    vision=vision,
                    language=_language_evidence(spec_pair, instruction_label),
                )
            )
    if not examples:
        raise ValueError("No calibration tasks use the requested monitor pair")
    return tuple(examples)


@dataclass(frozen=True)
class PairwiseProbeSelection:
    labels: tuple[str, str]
    rows: tuple[ProbeSelectionRow, ...]
    selected: ProbeSelectionRow


def select_pairwise_probes(
    dataset: ProbeDataset,
    specs: Mapping[int, SupportConflictSpec],
    *,
    alpha_candidates: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    temperature_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> tuple[dict[str, PairwiseProbe], tuple[PairwiseProbeSelection, ...]]:
    """Fit one binary monitor for each intervention-supported label pair."""

    pairs = tuple(
        dict.fromkeys(pair_key((spec.true_label, spec.distractor_label)) for spec in specs.values())
    )
    probes: dict[str, PairwiseProbe] = {}
    selections: list[PairwiseProbeSelection] = []
    for key in pairs:
        labels = cast(tuple[str, str], tuple(key.split("|")))
        subset = _pair_dataset(dataset, labels)
        rows: list[ProbeSelectionRow] = []
        for alpha in alpha_candidates:
            logits = leave_episode_out_logits(subset, alpha=alpha)
            accuracy = float((logits.argmax(dim=1) == subset.label_indices).double().mean())
            for temperature in temperature_candidates:
                examples = _pair_probe_examples(subset, logits, specs, temperature=temperature)
                calibration = calibrate_conflict_monitor(
                    examples,
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
                rows.append(ProbeSelectionRow(alpha, temperature, accuracy, calibration))
        selected = max(
            rows,
            key=lambda row: (
                -row.calibration.selected.aligned_false_trigger_rate,
                row.calibration.selected.conflict_recall,
                row.multiclass_accuracy,
                -row.calibration.selected.aligned_abstention_rate,
                -row.alpha,
                -row.temperature,
            ),
        )
        weight = fit_ridge_probe(
            subset.features, subset.label_indices, classes=2, alpha=selected.alpha
        )
        probes[key] = PairwiseProbe(
            labels=labels,
            weight=weight,
            alpha=selected.alpha,
            temperature=selected.temperature,
            divergence_threshold=selected.calibration.selected.divergence_threshold,
            confidence_floor=selected.calibration.selected.confidence_floor,
        )
        selections.append(PairwiseProbeSelection(labels, tuple(rows), selected))
    return probes, tuple(selections)


def collect_probe_dataset(
    policy: Any,
    adapter: Any,
    specs: tuple[SupportConflictSpec, ...],
    episode_indices: tuple[int, ...],
    *,
    representation_layer: int,
    representation_position: int,
    representation_pool: RepresentationPool,
    suite: str,
    resolution: int,
    simulator_seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> ProbeDataset:
    """Collect location-neutral frozen VLM keys from initial observations."""

    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    labels = tuple(sorted({spec.true_label for spec in specs}))
    features: list[Tensor] = []
    label_indices: list[int] = []
    task_ids: list[int] = []
    episodes: list[int] = []
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
                    raise ValueError("Native instruction differs from probe task spec")
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                batch = prepare_libero_batch(adapter, observation, _NEUTRAL_PROMPT)
                key = monitor_representation(
                    policy,
                    batch,
                    representation_layer=representation_layer,
                    representation_position=representation_position,
                    representation_pool=representation_pool,
                )
                features.append(key.cpu())
                label_indices.append(labels.index(spec.true_label))
                task_ids.append(spec.task_id)
                episodes.append(episode_index)
            finally:
                environment.close()
            if progress is not None:
                progress(spec.task_id, episode_index)
    return ProbeDataset(
        features=torch.cat(features),
        label_indices=torch.tensor(label_indices, dtype=torch.int64),
        task_ids=torch.tensor(task_ids, dtype=torch.int64),
        episode_indices=torch.tensor(episodes, dtype=torch.int64),
        labels=labels,
    )


def fit_visual_probe_monitor(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = tuple(range(10)),
    representation_layer: int = 16,
    representation_position: int = -2,
    representation_pool: RepresentationPool = "token",
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[VisualProbeBank, ProbeFitReport]:
    """Fit a visual monitor without using action outcomes or conflict prompts."""

    specs_by_task = {spec.task_id: spec for spec in support_conflict_specs()}
    specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    selected_specs = {spec.task_id: spec for spec in specs}
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    dataset = collect_probe_dataset(
        policy,
        adapter,
        specs,
        episode_indices,
        representation_layer=representation_layer,
        representation_position=representation_position,
        representation_pool=representation_pool,
        suite=suite,
        resolution=resolution,
        simulator_seed=simulator_seed,
        progress=progress,
    )
    bank, selection_rows, calibration = select_probe(dataset, selected_specs)
    bank = VisualProbeBank(
        **{
            **asdict(bank),
            "representation_layer": representation_layer,
            "representation_position": representation_position,
            "representation_pool": representation_pool,
        }
    )
    report = ProbeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        neutral_prompt=_NEUTRAL_PROMPT,
        representation_layer=representation_layer,
        representation_position=representation_position,
        representation_pool=representation_pool,
        training_task_ids=task_ids,
        training_episode_indices=episode_indices,
        labels=dataset.labels,
        samples=dataset.features.shape[0],
        selection_rows=selection_rows,
        selected_alpha=bank.alpha,
        selected_temperature=bank.temperature,
        calibration=calibration,
    )
    return bank, report


@dataclass(frozen=True)
class ProbeEvaluationReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    examples: tuple[LabeledMonitorExample, ...]
    operating_point: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            **{key: value for key, value in asdict(self).items() if key not in {"examples"}},
            "examples": [
                {
                    **{
                        key: value
                        for key, value in asdict(example).items()
                        if key not in {"vision", "language"}
                    },
                    "vision": asdict(example.vision),
                    "language": asdict(example.language),
                }
                for example in self.examples
            ],
        }


def evaluate_visual_probe_monitor(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = (15, 16, 17, 18, 19),
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> ProbeEvaluationReport:
    """Evaluate frozen probe and thresholds on disjoint development scenes."""

    bank = load_visual_probe_bank(bank_path)
    specs_by_task = {spec.task_id: spec for spec in support_conflict_specs()}
    specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    dataset = collect_probe_dataset(
        policy,
        adapter,
        specs,
        episode_indices,
        representation_layer=bank.representation_layer,
        representation_position=bank.representation_position,
        representation_pool=bank.representation_pool,
        suite=suite,
        resolution=resolution,
        simulator_seed=simulator_seed,
        progress=progress,
    )
    if dataset.labels != bank.labels:
        raise ValueError("Evaluation label vocabulary differs from probe bank")
    logits = probe_logits(dataset.features, bank.weight)
    examples = _calibration_examples(dataset, logits, specs_by_task, temperature=bank.temperature)
    point = evaluate_monitor_operating_point(
        examples,
        divergence_threshold=bank.divergence_threshold,
        confidence_floor=bank.confidence_floor,
    )
    return ProbeEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        task_ids=task_ids,
        episode_indices=episode_indices,
        examples=examples,
        operating_point=cast(dict[str, float | int], asdict(point)),
    )


@dataclass(frozen=True)
class PairwiseProbeFitReport:
    schema_version: int
    checkpoint: str
    suite: str
    neutral_prompt: str
    representation_layer: int
    representation_position: int
    representation_pool: RepresentationPool
    training_task_ids: tuple[int, ...]
    training_episode_indices: tuple[int, ...]
    samples: int
    selections: tuple[PairwiseProbeSelection, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "selections"},
            "selections": [
                {
                    "labels": selection.labels,
                    "selected_alpha": selection.selected.alpha,
                    "selected_temperature": selection.selected.temperature,
                    "selected_multiclass_accuracy": selection.selected.multiclass_accuracy,
                    "calibration": selection.selected.calibration.to_dict(),
                    "rows": [
                        {
                            "alpha": row.alpha,
                            "temperature": row.temperature,
                            "multiclass_accuracy": row.multiclass_accuracy,
                            "selected_operating_point": asdict(row.calibration.selected),
                        }
                        for row in selection.rows
                    ],
                }
                for selection in self.selections
            ],
        }


def fit_pairwise_visual_probe_monitor(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = tuple(range(10)),
    representation_layer: int = 16,
    representation_position: int = -2,
    representation_pool: RepresentationPool = "token",
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[PairwiseVisualProbeBank, PairwiseProbeFitReport]:
    """Fit transition-specific probes from one shared neutral-state scan."""

    specs_by_task = {spec.task_id: spec for spec in support_conflict_specs()}
    specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    selected_specs = {spec.task_id: spec for spec in specs}
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    dataset = collect_probe_dataset(
        policy,
        adapter,
        specs,
        episode_indices,
        representation_layer=representation_layer,
        representation_position=representation_position,
        representation_pool=representation_pool,
        suite=suite,
        resolution=resolution,
        simulator_seed=simulator_seed,
        progress=progress,
    )
    probes, selections = select_pairwise_probes(dataset, selected_specs)
    bank = PairwiseVisualProbeBank(
        representation_layer=representation_layer,
        representation_position=representation_position,
        representation_pool=representation_pool,
        neutral_prompt=_NEUTRAL_PROMPT,
        probes=probes,
        training_task_ids=task_ids,
        training_episode_indices=episode_indices,
    )
    report = PairwiseProbeFitReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        neutral_prompt=_NEUTRAL_PROMPT,
        representation_layer=representation_layer,
        representation_position=representation_position,
        representation_pool=representation_pool,
        training_task_ids=task_ids,
        training_episode_indices=episode_indices,
        samples=dataset.features.shape[0],
        selections=selections,
    )
    return bank, report


@dataclass(frozen=True)
class PairwiseProbeEvaluationReport:
    schema_version: int
    checkpoint: str
    bank_path: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    per_pair: dict[str, dict[str, float | int]]
    examples: tuple[LabeledMonitorExample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "examples"},
            "examples": [
                {
                    **{
                        key: value
                        for key, value in asdict(example).items()
                        if key not in {"vision", "language"}
                    },
                    "vision": asdict(example.vision),
                    "language": asdict(example.language),
                }
                for example in self.examples
            ],
        }


def evaluate_pairwise_visual_probe_monitor(
    checkpoint: str | Path,
    bank_path: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = (15, 16, 17, 18, 19),
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    simulator_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> PairwiseProbeEvaluationReport:
    """Evaluate every frozen pair-specific probe on disjoint development scenes."""

    bank = load_pairwise_visual_probe_bank(bank_path)
    specs_by_task = {spec.task_id: spec for spec in support_conflict_specs()}
    specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    dataset = collect_probe_dataset(
        policy,
        adapter,
        specs,
        episode_indices,
        representation_layer=bank.representation_layer,
        representation_position=bank.representation_position,
        representation_pool=bank.representation_pool,
        suite=suite,
        resolution=resolution,
        simulator_seed=simulator_seed,
        progress=progress,
    )
    examples: list[LabeledMonitorExample] = []
    per_pair: dict[str, dict[str, float | int]] = {}
    for key, probe in bank.probes.items():
        subset = _pair_dataset(dataset, probe.labels)
        logits = probe_logits(subset.features, probe.weight)
        pair_examples = _pair_probe_examples(
            subset, logits, specs_by_task, temperature=probe.temperature
        )
        point = evaluate_monitor_operating_point(
            pair_examples,
            divergence_threshold=probe.divergence_threshold,
            confidence_floor=probe.confidence_floor,
        )
        examples.extend(pair_examples)
        per_pair[key] = cast(dict[str, float | int], asdict(point))
    return PairwiseProbeEvaluationReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        bank_path=str(Path(bank_path).resolve()),
        task_ids=task_ids,
        episode_indices=episode_indices,
        per_pair=per_pair,
        examples=tuple(examples),
    )


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.probe_monitor_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    fit.add_argument("--episodes", type=_integers, default=tuple(range(10)))
    fit.add_argument("--representation-layer", type=int, default=16)
    fit.add_argument("--representation-position", type=int, default=-2)
    fit.add_argument(
        "--representation-pool",
        choices=("token", "image_mean", "camera_means", "image_tokens"),
        default="token",
    )
    fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    fit.add_argument("--bank-output", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    evaluate.add_argument("--episodes", type=_integers, default=(15, 16, 17, 18, 19))
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    evaluate.add_argument("--output", type=Path, required=True)
    pair_fit = subparsers.add_parser("fit-pairwise")
    pair_fit.add_argument("--checkpoint", type=Path, required=True)
    pair_fit.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    pair_fit.add_argument("--episodes", type=_integers, default=tuple(range(10)))
    pair_fit.add_argument("--representation-layer", type=int, default=16)
    pair_fit.add_argument("--representation-position", type=int, default=-2)
    pair_fit.add_argument(
        "--representation-pool",
        choices=("token", "image_mean", "camera_means", "image_tokens"),
        default="token",
    )
    pair_fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    pair_fit.add_argument("--bank-output", type=Path, required=True)
    pair_fit.add_argument("--output", type=Path, required=True)
    pair_evaluate = subparsers.add_parser("evaluate-pairwise")
    pair_evaluate.add_argument("--checkpoint", type=Path, required=True)
    pair_evaluate.add_argument("--bank", type=Path, required=True)
    pair_evaluate.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    pair_evaluate.add_argument("--episodes", type=_integers, default=(15, 16, 17, 18, 19))
    pair_evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    pair_evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(task: int, episode: int) -> None:
        print(f"[probe-monitor] task={task} episode={episode}", file=sys.stderr, flush=True)

    if args.command == "fit":
        bank, fit_report = fit_visual_probe_monitor(
            args.checkpoint,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            representation_layer=args.representation_layer,
            representation_position=args.representation_position,
            representation_pool=args.representation_pool,
            device=args.device,
            progress=progress,
        )
        save_visual_probe_bank(bank, args.bank_output)
        _write_json(args.output, fit_report.to_dict())
        return 0
    if args.command == "evaluate":
        evaluation_report = evaluate_visual_probe_monitor(
            args.checkpoint,
            args.bank,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            device=args.device,
            progress=progress,
        )
        _write_json(args.output, evaluation_report.to_dict())
        return 0
    if args.command == "fit-pairwise":
        pair_bank, pair_fit_report = fit_pairwise_visual_probe_monitor(
            args.checkpoint,
            task_ids=args.tasks,
            episode_indices=args.episodes,
            representation_layer=args.representation_layer,
            representation_position=args.representation_position,
            representation_pool=args.representation_pool,
            device=args.device,
            progress=progress,
        )
        save_pairwise_visual_probe_bank(pair_bank, args.bank_output)
        _write_json(args.output, pair_fit_report.to_dict())
        return 0
    pair_evaluation_report = evaluate_pairwise_visual_probe_monitor(
        args.checkpoint,
        args.bank,
        task_ids=args.tasks,
        episode_indices=args.episodes,
        device=args.device,
        progress=progress,
    )
    _write_json(args.output, pair_evaluation_report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
