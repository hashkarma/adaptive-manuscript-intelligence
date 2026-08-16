from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


STAGE5_ROUTING_VERSION = "0.2.0-stage5g-route-low-h-to-stage6"


@dataclass(frozen=True)
class Stage5RoutingPolicy:
    stage4_min_segmentation_readiness: float = 0.60
    htr_high_readiness: float = 0.75
    htr_intermediate_readiness: float = 0.50
    min_decoder_reliability_for_accept: float = 0.55
    min_cross_provider_agreement_for_accept: float = 0.75
    low_decoder_reliability: float = 0.25
    low_cross_provider_agreement: float = 0.50
    max_low_readiness_line_fraction_for_accept: float = 0.00
    max_low_agreement_line_fraction_for_accept: float = 0.20


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_readiness_artifact(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 5 readiness artifact is missing: {path}"
        )

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Stage 5 readiness artifact must be a JSON object.")

    if "page" not in data:
        raise ValueError("Stage 5 readiness artifact has no 'page' object.")

    return data


def _required_signal(evidence: Dict[str, Any], name: str) -> float:
    value = _safe_float(evidence.get(name))
    if value is None:
        raise ValueError(
            f"Required Stage 5 readiness evidence '{name}' is unavailable."
        )
    return _clip01(value)


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def evaluate_stage5_routing(
    readiness: Dict[str, Any],
    *,
    third_provider_available: bool = False,
    policy: Optional[Stage5RoutingPolicy] = None,
) -> Dict[str, Any]:
    """
    Stage 5G adaptive routing.

    Stage 5F owns H(p). Stage 5G only converts frozen evidence into a
    downstream orchestration decision.

    Critical policy change in v0.2:
      - healthy Stage 4 + low H(p) no longer routes directly to scholar review;
      - the page proceeds to Stage 6 in deep-reconstruction mode;
      - a materially different third HTR provider, if available, is retained as
        a later adaptive retry option after Stage 6 computes T(p).

    Human/scholar review is therefore not a Stage-5 terminal branch.
    """
    if policy is None:
        policy = Stage5RoutingPolicy()

    page = readiness.get("page", {})
    evidence = page.get("evidence", {})

    if not isinstance(page, dict) or not isinstance(evidence, dict):
        raise ValueError("Malformed Stage 5 readiness page/evidence structure.")

    S = _required_signal(evidence, "segmentation_readiness")

    H = _safe_float(page.get("htr_readiness_H_page"))
    if H is None:
        raise ValueError("Page-level HTR readiness H(p) is unavailable.")
    H = _clip01(H)

    completion = _required_signal(evidence, "completion")
    script_integrity = _required_signal(evidence, "script_integrity")
    decoder_reliability = _required_signal(evidence, "decoder_reliability")
    sequence_quality = _required_signal(evidence, "sequence_quality")
    agreement = _required_signal(evidence, "cross_provider_agreement")

    total_lines = _safe_int(page.get("total_lines"), 0)
    low_readiness_lines = _safe_int(page.get("low_readiness_lines"), 0)
    low_agreement_lines = _safe_int(page.get("low_agreement_lines"), 0)

    low_readiness_fraction = _fraction(low_readiness_lines, total_lines)
    low_agreement_fraction = _fraction(low_agreement_lines, total_lines)

    complaints: List[str] = []
    recommendations: List[str] = []
    hard_gates: List[str] = []

    status = "ok"
    next_action = "proceed_to_stage6_standard"
    decision = "provisionally_proceed_to_stage6"
    stage6_mode = "standard_semantic_processing"

    if S < policy.stage4_min_segmentation_readiness:
        status = "blocked"
        decision = "return_to_stage4"
        next_action = "retry_stage4_or_upstream"
        stage6_mode = None

        hard_gates.append("LOW_STAGE4_SEGMENTATION_READINESS")
        complaints.append("LOW_STAGE4_SEGMENTATION_READINESS")
        recommendations.append(
            "Do not spend additional HTR or semantic compute until Stage 4 "
            "segmentation readiness is repaired by an automatic Stage-4 or "
            "upstream retry."
        )

    elif completion < 1.0:
        status = "blocked"
        decision = "retry_or_switch_htr_provider"
        next_action = "retry_or_switch_htr_provider"
        stage6_mode = None

        hard_gates.append("INCOMPLETE_HTR_PROVIDER_COVERAGE")
        complaints.append("INCOMPLETE_HTR_PROVIDER_COVERAGE")
        recommendations.append(
            "Retry failed HTR inference or switch provider before downstream "
            "semantic processing."
        )

    else:
        if script_integrity < 0.90:
            complaints.append("LOW_SCRIPT_OR_TRANSLITERATION_INTEGRITY")

        if decoder_reliability < policy.low_decoder_reliability:
            complaints.append("LOW_DECODER_RELIABILITY")

        if agreement < policy.low_cross_provider_agreement:
            complaints.append("LOW_CROSS_PROVIDER_AGREEMENT")

        if low_agreement_fraction > (
            policy.max_low_agreement_line_fraction_for_accept
        ):
            complaints.append("PAGE_WIDE_LOW_CROSS_PROVIDER_AGREEMENT")

        if low_readiness_fraction > (
            policy.max_low_readiness_line_fraction_for_accept
        ):
            complaints.append("PAGE_CONTAINS_LOW_HTR_READINESS_LINES")

        if H < policy.htr_intermediate_readiness:
            status = "warning"
            decision = "proceed_to_stage6_deep_reconstruction"
            next_action = "proceed_to_stage6_deep_reconstruction"
            stage6_mode = "deep_semantic_reconstruction"

            recommendations.append(
                "Stage 4 is sufficiently ready but HTR evidence is weak. "
                "Proceed to Stage 6 using deep, evidence-constrained semantic "
                "reconstruction. Do not route directly to scholar review."
            )

            if third_provider_available:
                recommendations.append(
                    "A materially different third HTR provider is available. "
                    "Reserve it as an adaptive retry option if Stage 6 later "
                    "produces low T(p)."
                )

        elif H < policy.htr_high_readiness:
            status = "warning"
            decision = "proceed_to_stage6_enhanced_reconstruction"
            next_action = "proceed_to_stage6_enhanced_reconstruction"
            stage6_mode = "enhanced_semantic_reconstruction"

            recommendations.append(
                "HTR readiness is intermediate. Proceed to Stage 6 with "
                "enhanced candidate, morphology and contextual evidence rather "
                "than stopping inside Stage 5."
            )

        else:
            acceptance_failures: List[str] = []

            if decoder_reliability < policy.min_decoder_reliability_for_accept:
                acceptance_failures.append(
                    "DECODER_RELIABILITY_BELOW_ACCEPT_THRESHOLD"
                )

            if agreement < policy.min_cross_provider_agreement_for_accept:
                acceptance_failures.append(
                    "CROSS_PROVIDER_AGREEMENT_BELOW_ACCEPT_THRESHOLD"
                )

            if low_readiness_fraction > (
                policy.max_low_readiness_line_fraction_for_accept
            ):
                acceptance_failures.append("LOW_READINESS_LINES_PRESENT")

            if low_agreement_fraction > (
                policy.max_low_agreement_line_fraction_for_accept
            ):
                acceptance_failures.append("TOO_MANY_LOW_AGREEMENT_LINES")

            if acceptance_failures:
                status = "warning"
                decision = "proceed_to_stage6_enhanced_reconstruction"
                next_action = "proceed_to_stage6_enhanced_reconstruction"
                stage6_mode = "enhanced_semantic_reconstruction"

                hard_gates.extend(acceptance_failures)
                complaints.extend(acceptance_failures)

                recommendations.append(
                    "Numerical H(p) is high but one or more acceptance gates "
                    "failed. Proceed to Stage 6 in enhanced mode instead of "
                    "claiming trusted transcription."
                )

            else:
                status = "ok"
                decision = "provisionally_proceed_to_stage6"
                next_action = "proceed_to_stage6_standard"
                stage6_mode = "standard_semantic_processing"

                recommendations.append(
                    "HTR system readiness is high enough for provisional "
                    "downstream semantic processing. This is not scholarly "
                    "validation and does not replace CER/WER."
                )

    complaints = list(dict.fromkeys(complaints))
    hard_gates = list(dict.fromkeys(hard_gates))

    return {
        "layer": "layer5",
        "stage": "stage5_htr",
        "routing_version": STAGE5_ROUTING_VERSION,
        "status": status,
        "signals": {
            "S": round(S, 4),
            "H": round(H, 4),
            "interpretation": {
                "S": "Stage 4 segmentation readiness available to Stage 5.",
                "H": (
                    "HTR system readiness for downstream processing; "
                    "not recognition accuracy."
                ),
            },
        },
        "evidence": {
            "completion": round(completion, 4),
            "script_integrity": round(script_integrity, 4),
            "decoder_reliability": round(decoder_reliability, 4),
            "sequence_quality": round(sequence_quality, 4),
            "cross_provider_agreement": round(agreement, 4),
            "total_lines": total_lines,
            "low_readiness_lines": low_readiness_lines,
            "low_readiness_fraction": round(low_readiness_fraction, 4),
            "low_agreement_lines": low_agreement_lines,
            "low_agreement_fraction": round(low_agreement_fraction, 4),
        },
        "complaints": complaints,
        "hard_gates": hard_gates,
        "decision": decision,
        "next_action": next_action,
        "stage6_reconstruction_mode": stage6_mode,
        "recommendations": recommendations,
        "available_htr_provider_count": (
            3 if third_provider_available else 2
        ),
        "third_provider_available": third_provider_available,
        "third_provider_reserved_for_post_stage6_retry": bool(
            third_provider_available
            and S >= policy.stage4_min_segmentation_readiness
        ),
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_adaptive_retries_exhausted": True,
        "ground_truth": {
            "available": bool(page.get("ground_truth_available", False)),
            "cer": page.get("cer"),
            "wer": page.get("wer"),
            "recognition_accuracy_calibrated": bool(
                page.get("recognition_accuracy_calibrated", False)
            ),
        },
        "policy": asdict(policy),
        "source_readiness_version": readiness.get("readiness_version"),
        "audit_note": (
            "Stage 5G consumes Stage 5F evidence and never recalculates H(p), "
            "CER or WER. Low H(p) with healthy segmentation now proceeds to "
            "Stage 6 deep reconstruction. Human review is not a Stage-5 "
            "terminal route."
        ),
    }


def write_stage5_routing_artifact(
    path: str,
    result: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
