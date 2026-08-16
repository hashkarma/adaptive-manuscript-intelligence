from __future__ import annotations

from orchestration.stage5_routing import (
    Stage5RoutingPolicy,
    evaluate_stage5_routing,
)
from orchestration.stage6_routing import (
    Stage6AdaptiveRetryPolicy,
    evaluate_stage6_adaptive_routing,
)


def interpret_signal(value: float, good: float = 0.70, warn: float = 0.45) -> str:
    if value >= good:
        return "good"
    if value >= warn:
        return "warning"
    return "poor"


def route_decision(signals: dict) -> dict:
    """
    Pre-Stage-4 routing decision.

    Poor upstream readiness produces an automatic retry/upstream-repair
    decision. Scholar review is reserved for Stage 6G.
    """
    C = float(signals["C"])
    B = float(signals["B"])
    G = float(signals["G"])
    L = float(signals["L"])
    R = float(signals["R"])

    complaints: list[str] = []
    recommendations: list[str] = []
    advisories: list[str] = []

    if C < 0.45:
        complaints.append("Enhancement adequacy is low.")
        recommendations.append(
            "Repeat Research Stage 1 with stronger local enhancement or an "
            "alternate restoration preset."
        )

    if B > 0.60:
        complaints.append(
            "Estimated bleed-through / background interference remains high."
        )
        recommendations.append(
            "Use or strengthen dedicated recto-verso interference handling "
            "before recognition."
        )

    if G < 0.35:
        complaints.append(
            "Geometry / structural reliability is critically weak."
        )
        recommendations.append(
            "Retry upstream damage/layout processing before Stage 4."
        )
    elif G < 0.55:
        advisories.append(
            "Geometry / structural reliability is comparatively low; "
            "Research Stage 4 must validate whether reliable line "
            "segmentation is still possible."
        )

    if L < 0.45:
        complaints.append(
            "Layout readiness is low; text-bearing regions are not reliably "
            "separable."
        )
        recommendations.append(
            "Retry Research Stage 3 layout analysis before Stage 4."
        )

    hard_block = C < 0.35 or G < 0.35 or L < 0.35 or R < 0.45

    if hard_block:
        status = "poor"
        next_action = "retry_upstream_before_stage4"
        summary = (
            "The page is not reliably ready for automatic line segmentation; "
            "retry the relevant upstream stage(s)."
        )
    elif R >= 0.70:
        if G < 0.55 or B > 0.45:
            status = "warning"
            next_action = "proceed_to_stage4_validation"
            summary = (
                "Overall readiness is acceptable, but an upstream risk "
                "remains. Proceed to Research Stage 4 and use segmentation "
                "evidence as a validation gate."
            )
        else:
            status = "ok"
            next_action = "proceed_to_stage4"
            summary = (
                "The page is sufficiently ready for Research Stage 4 "
                "script-aware line segmentation."
            )
    else:
        status = "warning"
        next_action = "proceed_to_stage4_with_caution"
        summary = (
            "The page may continue to Research Stage 4, but segmentation "
            "must be treated as a validation step."
        )

    return {
        "overall_status": status,
        "next_action": next_action,
        "summary": summary,
        "complaints": complaints,
        "recommendations": recommendations,
        "advisories": advisories,
        "critical_signal_checks": {
            "C_minimum_met": C >= 0.45,
            "G_minimum_met": G >= 0.35,
            "L_minimum_met": L >= 0.45,
            "R_minimum_met": R >= 0.45,
        },
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6": True,
    }


def route_after_segmentation(signals: dict, layer4_status: str) -> dict:
    """
    Post-Stage-4 routing decision.

    Poor Stage-4 evidence triggers an automatic Stage-4/upstream retry, not
    scholar review.
    """
    C = float(signals["C"])
    B = float(signals["B"])
    G = float(signals["G"])
    L = float(signals["L"])
    R = float(signals["R"])
    S = float(signals["S"])

    complaints: list[str] = []
    recommendations: list[str] = []
    advisories: list[str] = []
    validation_trace: list[str] = []

    if G < 0.55 and S >= 0.75:
        validation_trace.append(
            "Research Stage 4 successfully validated usable line structure "
            "despite the comparatively low upstream G(p) signal."
        )
    elif G < 0.55:
        validation_trace.append(
            "The upstream structural concern remains unresolved because "
            "Stage 4 segmentation readiness is not strong enough."
        )
    else:
        validation_trace.append(
            "Research Stage 4 confirmed the upstream page-readiness assessment."
        )

    if B > 0.45:
        advisories.append(
            "Bleed-through severity remains provisional and should be "
            "revisited when a dedicated interference stage is introduced."
        )

    if layer4_status == "poor" or S < 0.55:
        status = "poor"
        next_action = "retry_stage4_or_upstream"
        summary = (
            "Line segmentation is not reliable enough for HTR. Retry Stage 4 "
            "or the relevant upstream preparation automatically."
        )
        complaints.append("Stage 4 segmentation readiness is too low.")
        recommendations.append(
            "Retry Stage 4 with safer boundaries/peak settings and inspect "
            "upstream layout evidence if the retry remains poor."
        )
    elif layer4_status == "warning" or S < 0.75:
        status = "warning"
        next_action = "retry_stage4"
        summary = (
            "Stage 4 produced a usable diagnostic result, but segmentation "
            "should be retried before HTR."
        )
        complaints.append("Stage 4 segmentation requires another attempt.")
        recommendations.append(
            "Retry Stage 4 and compare line crops, uncertainty and "
            "losslessness before continuing."
        )
    elif R < 0.45:
        status = "poor"
        next_action = "retry_upstream_before_htr"
        summary = (
            "Segmentation is strong, but upstream page readiness remains too "
            "weak for unattended HTR."
        )
        complaints.append("Upstream readiness remains below the minimum gate.")
        recommendations.append(
            "Retry restoration, damage or layout processing as indicated by "
            "C/B/G/L before HTR."
        )
    else:
        status = "ok"
        next_action = "proceed_to_htr"
        summary = (
            "Stage 4 has validated the page as suitable for Research Stage 5 "
            "sequence-based HTR."
        )

    return {
        "overall_status": status,
        "next_action": next_action,
        "summary": summary,
        "complaints": complaints,
        "recommendations": recommendations,
        "advisories": advisories,
        "validation_trace": validation_trace,
        "critical_signal_checks": {
            "R_minimum_met": R >= 0.45,
            "S_htr_gate_met": S >= 0.75,
            "stage4_not_poor": layer4_status != "poor",
        },
        "pipeline_context": {
            "C": C,
            "B": B,
            "G": G,
            "L": L,
            "R": R,
            "S": S,
        },
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6": True,
    }


def route_after_htr(
    stage5_readiness: dict,
    *,
    third_provider_available: bool = False,
    policy: Stage5RoutingPolicy | None = None,
) -> dict:
    """Low H(p) with healthy S(p) proceeds to Stage 6 deep reconstruction."""
    return evaluate_stage5_routing(
        stage5_readiness,
        third_provider_available=third_provider_available,
        policy=policy,
    )


def route_after_semantic_trust(
    stage6_trust: dict,
    stage6_reconstruction: dict,
    stage5_readiness: dict,
    layer4_report: dict,
    *,
    third_provider_available: bool = False,
    alternate_visual_htr_available: bool = False,
    expanded_context_retry_available: bool = False,
    retry_state: dict | None = None,
    policy: Stage6AdaptiveRetryPolicy | None = None,
) -> dict:
    """
    Post-Stage-6 adaptive routing.

    Stage 6G is the first policy point allowed to authorize scholar review
    after Stage 6 and machine-retry exhaustion.
    """
    return evaluate_stage6_adaptive_routing(
        stage6_trust,
        stage6_reconstruction,
        stage5_readiness,
        layer4_report,
        third_provider_available=third_provider_available,
        alternate_visual_htr_available=alternate_visual_htr_available,
        expanded_context_retry_available=expanded_context_retry_available,
        retry_state=retry_state,
        policy=policy,
    )
