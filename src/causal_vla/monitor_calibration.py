"""Development-only calibration for categorical visual-language conflict monitors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from causal_vla.routing import ConflictDetector, EvidenceDistribution

ExpectedMonitorStatus = Literal["aligned", "conflict"]


@dataclass(frozen=True)
class LabeledMonitorExample:
    """Frozen modality evidence with a known aligned/conflict label."""

    task_id: int
    episode_index: int
    conflict_type: str
    expected_status: ExpectedMonitorStatus
    true_visual_label: str
    vision: EvidenceDistribution
    language: EvidenceDistribution


@dataclass(frozen=True)
class MonitorOperatingPoint:
    """Confusion counts and abstention rates for one detector threshold pair."""

    divergence_threshold: float
    confidence_floor: float
    aligned_examples: int
    conflict_examples: int
    correct_conflict_triggers: int
    harmful_aligned_triggers: int
    conflict_abstentions: int
    aligned_abstentions: int
    conflict_recall: float
    aligned_false_trigger_rate: float
    conflict_abstention_rate: float
    aligned_abstention_rate: float


@dataclass(frozen=True)
class MonitorCalibration:
    """Selected operating point and the complete development threshold scan."""

    max_aligned_false_trigger_rate: float
    selected: MonitorOperatingPoint
    operating_points: tuple[MonitorOperatingPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "max_aligned_false_trigger_rate": self.max_aligned_false_trigger_rate,
            "selected": asdict(self.selected),
            "operating_points": [asdict(point) for point in self.operating_points],
        }


def evaluate_monitor_operating_point(
    examples: tuple[LabeledMonitorExample, ...],
    *,
    divergence_threshold: float,
    confidence_floor: float,
) -> MonitorOperatingPoint:
    """Evaluate a detector without allowing an incorrect visual target to count."""

    if not examples:
        raise ValueError("Monitor calibration requires labeled examples")
    detector = ConflictDetector(
        divergence_threshold=divergence_threshold,
        confidence_floor=confidence_floor,
    )
    aligned = sum(example.expected_status == "aligned" for example in examples)
    conflicts = len(examples) - aligned
    if not aligned or not conflicts:
        raise ValueError("Calibration requires both aligned and conflict examples")
    correct_triggers = 0
    harmful_triggers = 0
    conflict_abstentions = 0
    aligned_abstentions = 0
    for example in examples:
        assessment = detector.assess(example.vision, example.language)
        target_is_correct = assessment.vision_label == example.true_visual_label
        triggered = assessment.status == "conflict"
        if example.expected_status == "conflict":
            correct_triggers += int(triggered and target_is_correct)
            conflict_abstentions += int(assessment.status == "uncertain")
        else:
            harmful_triggers += int(triggered)
            aligned_abstentions += int(assessment.status == "uncertain")
    return MonitorOperatingPoint(
        divergence_threshold=divergence_threshold,
        confidence_floor=confidence_floor,
        aligned_examples=aligned,
        conflict_examples=conflicts,
        correct_conflict_triggers=correct_triggers,
        harmful_aligned_triggers=harmful_triggers,
        conflict_abstentions=conflict_abstentions,
        aligned_abstentions=aligned_abstentions,
        conflict_recall=correct_triggers / conflicts,
        aligned_false_trigger_rate=harmful_triggers / aligned,
        conflict_abstention_rate=conflict_abstentions / conflicts,
        aligned_abstention_rate=aligned_abstentions / aligned,
    )


def calibrate_conflict_monitor(
    examples: tuple[LabeledMonitorExample, ...],
    *,
    divergence_candidates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
    confidence_candidates: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
    max_aligned_false_trigger_rate: float = 0.0,
) -> MonitorCalibration:
    """Choose maximum conflict recall subject to a no-harm false-trigger budget."""

    if not divergence_candidates or not confidence_candidates:
        raise ValueError("Monitor threshold candidate sets must be nonempty")
    if not 0 <= max_aligned_false_trigger_rate <= 1:
        raise ValueError("False-trigger budget must lie in [0, 1]")
    points = tuple(
        evaluate_monitor_operating_point(
            examples,
            divergence_threshold=divergence,
            confidence_floor=confidence,
        )
        for divergence in divergence_candidates
        for confidence in confidence_candidates
    )
    feasible = tuple(
        point
        for point in points
        if point.aligned_false_trigger_rate <= max_aligned_false_trigger_rate
    )
    candidates = feasible or points
    selected = max(
        candidates,
        key=lambda point: (
            -point.aligned_false_trigger_rate,
            point.conflict_recall,
            -point.aligned_abstention_rate,
            -point.conflict_abstention_rate,
            point.confidence_floor,
            point.divergence_threshold,
        ),
    )
    return MonitorCalibration(
        max_aligned_false_trigger_rate=max_aligned_false_trigger_rate,
        selected=selected,
        operating_points=points,
    )
