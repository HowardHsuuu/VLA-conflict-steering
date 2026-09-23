from __future__ import annotations

import pytest

from causal_vla.routing import (
    AbstainPolicy,
    ConflictDetector,
    EvidenceDistribution,
    GroundedInternalSteeringPolicy,
    GroundedPromptCorrectionPolicy,
    jensen_shannon_divergence,
)


def evidence(source: str, first: float, second: float) -> EvidenceDistribution:
    return EvidenceDistribution(  # type: ignore[arg-type]
        source=source,
        proposition="goal_destination",
        labels=("cabinet", "stove"),
        probabilities=(first, second),
    )


def test_evidence_is_normalized_and_label_order_is_aligned() -> None:
    first = evidence("vision", 8, 2)
    second = EvidenceDistribution("language", "goal_destination", ("stove", "cabinet"), (2, 8))
    assert first.probabilities == pytest.approx((0.8, 0.2))
    assert jensen_shannon_divergence(first, second) == pytest.approx(0.0)


def test_aligned_modalities_follow_without_intervention() -> None:
    assessment = ConflictDetector().assess(
        evidence("vision", 0.9, 0.1),
        evidence("language", 0.85, 0.15),
    )
    decision = AbstainPolicy().decide(assessment, intervention_reachable=False)
    assert assessment.status == "aligned"
    assert decision.action == "follow"
    assert decision.target_label == "cabinet"


def test_high_confidence_visual_conflict_can_trigger_reachable_steering() -> None:
    assessment = ConflictDetector().assess(
        evidence("vision", 0.97, 0.03),
        evidence("language", 0.20, 0.80),
    )
    decision = AbstainPolicy().decide(assessment, intervention_reachable=True)
    assert assessment.status == "conflict"
    assert decision.action == "steer_to_vision"
    assert decision.target_label == "cabinet"


def test_ambiguous_visual_scene_abstains() -> None:
    assessment = ConflictDetector().assess(
        evidence("vision", 0.5, 0.5),
        evidence("language", 0.05, 0.95),
    )
    decision = AbstainPolicy().decide(assessment, intervention_reachable=True)
    assert assessment.status == "uncertain"
    assert decision.action == "abstain"


def test_unreachable_or_non_dominant_conflict_abstains() -> None:
    detector = ConflictDetector(divergence_threshold=0.1)
    reachable = detector.assess(
        evidence("vision", 0.90, 0.10),
        evidence("language", 0.25, 0.75),
    )
    assert (
        AbstainPolicy().decide(reachable, intervention_reachable=False).reason
        == "intervention_not_action_reachable"
    )

    non_dominant = detector.assess(
        evidence("vision", 0.75, 0.25),
        evidence("language", 0.30, 0.70),
    )
    assert (
        AbstainPolicy(dominance_margin=0.1).decide(non_dominant, intervention_reachable=True).reason
        == "conflict_without_dominant_evidence"
    )


def test_different_propositions_cannot_be_compared() -> None:
    vision = EvidenceDistribution("vision", "current_location", ("cabinet", "stove"), (0.9, 0.1))
    language = evidence("language", 0.1, 0.9)
    assessment = ConflictDetector().assess(vision, language)

    assert assessment.status == "uncertain"
    assert assessment.reason == "proposition_mismatch"
    assert assessment.divergence is None
    assert AbstainPolicy().decide(assessment, intervention_reachable=True).action == "abstain"


def test_grounded_prompt_policy_rewrites_confident_factual_conflict() -> None:
    assessment = ConflictDetector().assess(
        evidence("vision", 0.76, 0.24),
        evidence("language", 0.02, 0.98),
    )
    decision = GroundedPromptCorrectionPolicy(vision_confidence_floor=0.70).decide(
        assessment,
        prompts_by_label={"cabinet": "use cabinet", "stove": "use stove"},
    )

    assert decision.action == "rewrite"
    assert decision.target_label == "cabinet"
    assert decision.prompt == "use cabinet"
    assert decision.reason == "grounded_visual_conflict"


def test_grounded_prompt_policy_abstains_without_grounded_visual_confidence() -> None:
    assessment = ConflictDetector(confidence_floor=0.5).assess(
        evidence("vision", 0.60, 0.40),
        evidence("language", 0.02, 0.98),
    )
    decision = GroundedPromptCorrectionPolicy(vision_confidence_floor=0.70).decide(
        assessment,
        prompts_by_label={"cabinet": "use cabinet", "stove": "use stove"},
    )

    assert decision.action == "abstain"
    assert decision.reason == "visual_evidence_below_correction_floor"


def test_grounded_internal_policy_selects_signed_cached_direction() -> None:
    detector = ConflictDetector(confidence_floor=0.5)
    forward = detector.assess(
        evidence("vision", 0.20, 0.80),
        evidence("language", 0.90, 0.10),
    )
    reverse = detector.assess(
        evidence("vision", 0.80, 0.20),
        evidence("language", 0.10, 0.90),
    )
    policy = GroundedInternalSteeringPolicy(vision_confidence_floor=0.70)

    forward_decision = policy.decide(forward, direction_labels=("cabinet", "stove"))
    reverse_decision = policy.decide(reverse, direction_labels=("cabinet", "stove"))

    assert forward_decision.action == "steer"
    assert forward_decision.direction_sign == 1
    assert reverse_decision.action == "steer"
    assert reverse_decision.direction_sign == -1


def test_grounded_internal_policy_abstains_below_steering_floor() -> None:
    assessment = ConflictDetector(
        divergence_threshold=0.05,
        confidence_floor=0.5,
    ).assess(
        evidence("vision", 0.40, 0.60),
        evidence("language", 0.90, 0.10),
    )
    assert assessment.status == "conflict"
    decision = GroundedInternalSteeringPolicy(vision_confidence_floor=0.70).decide(
        assessment, direction_labels=("cabinet", "stove")
    )

    assert decision.action == "abstain"
    assert decision.reason == "visual_evidence_below_steering_floor"


@pytest.mark.parametrize(
    "distribution",
    [
        (("cabinet",), ()),
        (("cabinet", "cabinet"), (0.5, 0.5)),
        (("cabinet",), (-1.0,)),
        (("cabinet",), (0.0,)),
    ],
)
def test_invalid_evidence_is_rejected(
    distribution: tuple[tuple[str, ...], tuple[float, ...]],
) -> None:
    with pytest.raises(ValueError):
        EvidenceDistribution("vision", "goal_destination", *distribution)
