"""Calibrated multimodal conflict detection and conservative action routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

EvidenceSource = Literal["vision", "language"]
ConflictStatus = Literal["aligned", "conflict", "uncertain"]
RouteAction = Literal["follow", "steer_to_vision", "abstain"]
CorrectionAction = Literal["keep", "rewrite", "abstain"]
InternalSteeringAction = Literal["follow", "steer", "abstain"]
TemporalGateState = Literal["armed", "active", "bypassed", "released"]
TemporalGateEvent = Literal["none", "trigger", "bypass", "release"]


@dataclass(frozen=True)
class EvidenceDistribution:
    """A normalized categorical belief produced by one modality."""

    source: EvidenceSource
    proposition: str
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("Evidence must contain at least one label")
        if len(self.labels) != len(self.probabilities):
            raise ValueError("Labels and probabilities must have equal length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Evidence labels must be unique")
        if any(not math.isfinite(value) or value < 0 for value in self.probabilities):
            raise ValueError("Evidence probabilities must be finite and nonnegative")
        total = sum(self.probabilities)
        if total <= 0:
            raise ValueError("Evidence probabilities must have positive mass")
        object.__setattr__(
            self,
            "probabilities",
            tuple(value / total for value in self.probabilities),
        )

    @property
    def top_label(self) -> str:
        """Return the maximum-probability label with deterministic tie breaking."""

        index = max(
            range(len(self.labels)),
            key=lambda candidate: (self.probabilities[candidate], -candidate),
        )
        return self.labels[index]

    @property
    def confidence(self) -> float:
        """Return the largest categorical probability."""

        return max(self.probabilities)

    def aligned_probabilities(self, labels: tuple[str, ...]) -> tuple[float, ...]:
        """Return probabilities in a requested label order."""

        if set(labels) != set(self.labels):
            raise ValueError("Modalities must describe the same label set")
        lookup = dict(zip(self.labels, self.probabilities, strict=True))
        return tuple(lookup[label] for label in labels)


def jensen_shannon_divergence(
    first: EvidenceDistribution,
    second: EvidenceDistribution,
) -> float:
    """Return base-2 Jensen-Shannon divergence in the closed interval [0, 1]."""

    first_values = first.probabilities
    second_values = second.aligned_probabilities(first.labels)
    midpoint = tuple(
        (left + right) / 2 for left, right in zip(first_values, second_values, strict=True)
    )

    def kl(values: tuple[float, ...]) -> float:
        return sum(
            value * math.log2(value / center)
            for value, center in zip(values, midpoint, strict=True)
            if value > 0
        )

    return (kl(first_values) + kl(second_values)) / 2


@dataclass(frozen=True)
class ConflictAssessment:
    """Auditable result of comparing visual and language evidence."""

    status: ConflictStatus
    reason: str
    divergence: float | None
    vision_label: str
    vision_confidence: float
    language_label: str
    language_confidence: float


@dataclass(frozen=True)
class ConflictDetector:
    """Detect categorical modality conflict with an explicit uncertainty region."""

    divergence_threshold: float = 0.25
    confidence_floor: float = 0.65

    def __post_init__(self) -> None:
        if not 0 <= self.divergence_threshold <= 1:
            raise ValueError("divergence_threshold must lie in [0, 1]")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must lie in [0, 1]")

    def assess(
        self,
        vision: EvidenceDistribution,
        language: EvidenceDistribution,
    ) -> ConflictAssessment:
        """Compare evidence without assuming either modality is authoritative."""

        if vision.source != "vision" or language.source != "language":
            raise ValueError("assess expects vision evidence followed by language evidence")
        if vision.proposition != language.proposition:
            return ConflictAssessment(
                status="uncertain",
                reason="proposition_mismatch",
                divergence=None,
                vision_label=vision.top_label,
                vision_confidence=vision.confidence,
                language_label=language.top_label,
                language_confidence=language.confidence,
            )
        divergence = jensen_shannon_divergence(vision, language)
        confident = (
            vision.confidence >= self.confidence_floor
            and language.confidence >= self.confidence_floor
        )
        if not confident:
            status: ConflictStatus = "uncertain"
            reason = "low_modality_confidence"
        elif vision.top_label == language.top_label and divergence < self.divergence_threshold:
            status = "aligned"
            reason = "modalities_aligned"
        elif vision.top_label != language.top_label and divergence >= self.divergence_threshold:
            status = "conflict"
            reason = "modalities_conflict"
        else:
            status = "uncertain"
            reason = "disagreement_below_threshold"
        return ConflictAssessment(
            status=status,
            reason=reason,
            divergence=divergence,
            vision_label=vision.top_label,
            vision_confidence=vision.confidence,
            language_label=language.top_label,
            language_confidence=language.confidence,
        )


@dataclass(frozen=True)
class RoutingDecision:
    """A safe action and machine-readable reason for a monitored decision."""

    action: RouteAction
    target_label: str | None
    reason: str


@dataclass(frozen=True)
class AbstainPolicy:
    """Permit steering only when evidence and causal reachability justify it."""

    dominance_margin: float = 0.15

    def __post_init__(self) -> None:
        if not 0 <= self.dominance_margin <= 1:
            raise ValueError("dominance_margin must lie in [0, 1]")

    def decide(
        self,
        assessment: ConflictAssessment,
        *,
        intervention_reachable: bool,
    ) -> RoutingDecision:
        """Route, steer, or safely abstain from the assessed model decision."""

        if assessment.status == "aligned":
            return RoutingDecision("follow", assessment.language_label, "modalities_aligned")
        if assessment.status == "uncertain":
            return RoutingDecision("abstain", None, assessment.reason)
        if not intervention_reachable:
            return RoutingDecision("abstain", None, "intervention_not_action_reachable")
        advantage = assessment.vision_confidence - assessment.language_confidence
        if advantage >= self.dominance_margin:
            return RoutingDecision(
                "steer_to_vision",
                assessment.vision_label,
                "vision_confidently_dominates_language",
            )
        return RoutingDecision("abstain", None, "conflict_without_dominant_evidence")


@dataclass(frozen=True)
class PromptCorrectionDecision:
    """Auditable selection of a grounded instruction or a safe abstention."""

    action: CorrectionAction
    target_label: str | None
    prompt: str | None
    reason: str


@dataclass(frozen=True)
class GroundedPromptCorrectionPolicy:
    """Rewrite a factual instruction only from a closed set of visual labels.

    This policy is appropriate only when both modalities describe the same
    externally observable proposition. It intentionally does not infer latent user
    intent, and it never generates an unconstrained instruction.
    """

    vision_confidence_floor: float = 0.70

    def __post_init__(self) -> None:
        if not 0 <= self.vision_confidence_floor <= 1:
            raise ValueError("vision_confidence_floor must lie in [0, 1]")

    def decide(
        self,
        assessment: ConflictAssessment,
        *,
        prompts_by_label: dict[str, str],
    ) -> PromptCorrectionDecision:
        """Keep, rewrite, or abstain using an explicit label-to-prompt allowlist."""

        if assessment.status == "uncertain":
            return PromptCorrectionDecision("abstain", None, None, assessment.reason)
        if assessment.status == "aligned":
            prompt = prompts_by_label.get(assessment.language_label)
            if prompt is None:
                return PromptCorrectionDecision(
                    "abstain", None, None, "language_label_not_grounded"
                )
            return PromptCorrectionDecision(
                "keep", assessment.language_label, prompt, "modalities_aligned"
            )
        if assessment.vision_confidence < self.vision_confidence_floor:
            return PromptCorrectionDecision(
                "abstain", None, None, "visual_evidence_below_correction_floor"
            )
        prompt = prompts_by_label.get(assessment.vision_label)
        if prompt is None:
            return PromptCorrectionDecision("abstain", None, None, "vision_label_not_grounded")
        return PromptCorrectionDecision(
            "rewrite",
            assessment.vision_label,
            prompt,
            "grounded_visual_conflict",
        )


@dataclass(frozen=True)
class InternalSteeringDecision:
    """Auditable choice to follow, apply a signed direction, or abstain."""

    action: InternalSteeringAction
    source_label: str | None
    target_label: str | None
    direction_sign: int | None
    reason: str


@dataclass(frozen=True)
class GroundedInternalSteeringPolicy:
    """Select a signed cached direction for an observable factual conflict."""

    vision_confidence_floor: float = 0.70

    def __post_init__(self) -> None:
        if not 0 <= self.vision_confidence_floor <= 1:
            raise ValueError("vision_confidence_floor must lie in [0, 1]")

    def decide(
        self,
        assessment: ConflictAssessment,
        *,
        direction_labels: tuple[str, str],
    ) -> InternalSteeringDecision:
        """Map a same-proposition assessment to a learned direction and sign."""

        base_label, target_label = direction_labels
        if base_label == target_label:
            raise ValueError("Direction labels must be distinct")
        if assessment.status == "uncertain":
            return InternalSteeringDecision("abstain", None, None, None, assessment.reason)
        if assessment.status == "aligned":
            return InternalSteeringDecision(
                "follow",
                assessment.language_label,
                assessment.language_label,
                0,
                "modalities_aligned",
            )
        if assessment.vision_confidence < self.vision_confidence_floor:
            return InternalSteeringDecision(
                "abstain",
                assessment.language_label,
                assessment.vision_label,
                None,
                "visual_evidence_below_steering_floor",
            )
        transition = (assessment.language_label, assessment.vision_label)
        if transition == (base_label, target_label):
            sign = 1
        elif transition == (target_label, base_label):
            sign = -1
        else:
            return InternalSteeringDecision(
                "abstain",
                assessment.language_label,
                assessment.vision_label,
                None,
                "unsupported_label_transition",
            )
        return InternalSteeringDecision(
            "steer",
            assessment.language_label,
            assessment.vision_label,
            sign,
            "grounded_visual_conflict",
        )


@dataclass(frozen=True)
class TemporalGateDecision:
    """Observable state transition for one temporal monitor update."""

    state: TemporalGateState
    steering_active: bool
    event: TemporalGateEvent
    reason: str
    conflict_streak: int
    clear_streak: int
    checks: int


@dataclass
class TemporalConflictGate:
    """One-shot conflict trigger with hysteretic release.

    Source-location statements are only valid before manipulation begins.  The gate
    therefore permits a trigger while it is armed, permanently bypasses steering
    after a confident aligned assessment, and never re-arms after release.  Requiring
    consecutive clear assessments prevents a single noisy frame from releasing an
    active intervention.
    """

    trigger_patience: int = 1
    release_patience: int = 2
    arm_timeout: int = 3
    state: TemporalGateState = "armed"
    conflict_streak: int = 0
    clear_streak: int = 0
    checks: int = 0

    def __post_init__(self) -> None:
        if self.trigger_patience <= 0:
            raise ValueError("trigger_patience must be positive")
        if self.release_patience <= 0:
            raise ValueError("release_patience must be positive")
        if self.arm_timeout <= 0:
            raise ValueError("arm_timeout must be positive")

    def update(self, assessment: ConflictAssessment) -> TemporalGateDecision:
        """Advance the gate from one calibrated conflict assessment."""

        self.checks += 1
        event: TemporalGateEvent = "none"
        reason = "terminal_gate_state"
        if self.state == "armed":
            if assessment.status == "conflict":
                self.conflict_streak += 1
                self.clear_streak = 0
                reason = "awaiting_trigger_patience"
                if self.conflict_streak >= self.trigger_patience:
                    self.state = "active"
                    event = "trigger"
                    reason = "conflict_triggered"
            elif assessment.status == "aligned":
                self.state = "bypassed"
                self.conflict_streak = 0
                event = "bypass"
                reason = "initial_modalities_aligned"
            else:
                self.conflict_streak = 0
                reason = "initial_assessment_uncertain"
                if self.checks >= self.arm_timeout:
                    self.state = "bypassed"
                    event = "bypass"
                    reason = "arm_timeout_without_conflict"
        elif self.state == "active":
            if assessment.status == "conflict":
                self.clear_streak = 0
                reason = "conflict_persists"
            else:
                self.clear_streak += 1
                reason = "awaiting_release_patience"
                if self.clear_streak >= self.release_patience:
                    self.state = "released"
                    event = "release"
                    reason = "conflict_cleared"
        return TemporalGateDecision(
            state=self.state,
            steering_active=self.state == "active",
            event=event,
            reason=reason,
            conflict_streak=self.conflict_streak,
            clear_streak=self.clear_streak,
            checks=self.checks,
        )

    def release_for_phase_transition(self, reason: str) -> TemporalGateDecision:
        """Permanently release an active gate after a trusted task-phase transition."""

        if not reason:
            raise ValueError("Phase-transition release reason must be nonempty")
        if self.state != "active":
            raise RuntimeError("Only an active temporal gate can be externally released")
        self.state = "released"
        self.clear_streak = self.release_patience
        return TemporalGateDecision(
            state=self.state,
            steering_active=False,
            event="release",
            reason=reason,
            conflict_streak=self.conflict_streak,
            clear_streak=self.clear_streak,
            checks=self.checks,
        )
