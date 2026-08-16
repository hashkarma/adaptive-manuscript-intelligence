from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


STAGE6_ROUTING_VERSION = "0.1.0-stage6g-adaptive-retry-controller"


@dataclass(frozen=True)
class Stage6AdaptiveRetryPolicy:
    stage4_min_segmentation_readiness: float = 0.60
    htr_low_readiness: float = 0.50
    translation_trust_threshold: float = 0.65
    high_trust_threshold: float = 0.80
    max_machine_retry_attempts: int = 3


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _page_T(stage6_trust: Dict[str, Any]) -> float:
    metrics = stage6_trust.get("metrics", {}) or {}
    value = metrics.get("page_T")

    if value is None:
        value = stage6_trust.get("page_T")

    if value is None:
        raise ValueError(
            "Stage 6F artifact does not contain page-level T(p)."
        )

    return _clip01(float(value))


def _page_H(stage5_readiness: Dict[str, Any]) -> float:
    page = stage5_readiness.get("page", {}) or {}
    value = page.get("htr_readiness_H_page")

    if value is None:
        raise ValueError(
            "Stage 5 readiness artifact does not contain page-level H(p)."
        )

    return _clip01(float(value))


def _stage4_S(
    layer4_report: Dict[str, Any],
    stage5_readiness: Dict[str, Any],
) -> float:
    layer4_signals = layer4_report.get("signals", {}) or {}
    value = layer4_signals.get("S")

    if value is None:
        page = stage5_readiness.get("page", {}) or {}
        evidence = page.get("evidence", {}) or {}
        value = evidence.get("segmentation_readiness")

    if value is None:
        raise ValueError(
            "Neither Layer 4 report nor Stage 5 readiness contains S(p)."
        )

    return _clip01(float(value))


def _stage6_completion(
    stage6_reconstruction: Dict[str, Any],
) -> Dict[str, int]:
    metrics = stage6_reconstruction.get("metrics", {}) or {}

    return {
        "num_lines": _safe_int(metrics.get("num_lines"), 0),
        "reconstructed_lines": _safe_int(
            metrics.get("reconstructed_lines"), 0
        ),
        "partially_supported_lines": _safe_int(
            metrics.get("partially_supported_lines"), 0
        ),
        "abstained_lines": _safe_int(
            metrics.get("abstained_lines"), 0
        ),
        "normalized_lines": _safe_int(
            metrics.get("normalized_lines"), 0
        ),
    }


def evaluate_stage6_adaptive_routing(
    stage6_trust: Dict[str, Any],
    stage6_reconstruction: Dict[str, Any],
    stage5_readiness: Dict[str, Any],
    layer4_report: Dict[str, Any],
    *,
    third_provider_available: bool = False,
    alternate_visual_htr_available: bool = False,
    expanded_context_retry_available: bool = False,
    retry_state: Optional[Dict[str, Any]] = None,
    policy: Optional[Stage6AdaptiveRetryPolicy] = None,
) -> Dict[str, Any]:
    """
    Stage 6G adaptive retry controller.

    This is the first stage allowed to authorize scholar review.

    Scholar review is allowed only when Stage 6 has completed and either the
    retry budget is exhausted or no additional executable machine strategy is
    currently available.
    """
    if policy is None:
        policy = Stage6AdaptiveRetryPolicy()

    if retry_state is None:
        retry_state = {}

    T = _page_T(stage6_trust)
    H = _page_H(stage5_readiness)
    S = _stage4_S(layer4_report, stage5_readiness)

    trust_metrics = stage6_trust.get("metrics", {}) or {}
    completion = _stage6_completion(stage6_reconstruction)

    retry_attempts = _safe_int(retry_state.get("attempt_count"), 0)

    attempted_actions = [
        str(value)
        for value in retry_state.get("attempted_actions", [])
        if value
    ]

    unresolved_lines = _safe_int(
        trust_metrics.get("unresolved_lines"),
        completion["abstained_lines"],
    )

    translation_eligible_lines = _safe_int(
        trust_metrics.get("translation_eligible_lines"),
        0,
    )

    normalized_complete = (
        completion["num_lines"] > 0
        and completion["normalized_lines"] == completion["num_lines"]
    )

    complaints: List[str] = []
    recommendations: List[str] = []
    validation_trace: List[str] = []

    if S >= 0.75:
        validation_trace.append(
            "Stage 4 segmentation is strong; do not bounce the page back to "
            "segmentation merely because H(p) or T(p) is low."
        )
    else:
        validation_trace.append(
            "Stage 4 segmentation is below the strong validation range and "
            "may remain an upstream contributor to downstream failure."
        )

    validation_trace.append(
        f"Stage 5 H(p)={H:.4f}; this is HTR system readiness, not accuracy."
    )
    validation_trace.append(
        f"Stage 6F T(p)={T:.4f}; this is semantic/transcription trust/readiness, "
        "not a calibrated probability."
    )

    candidate_actions: List[Dict[str, Any]] = []

    if third_provider_available:
        candidate_actions.append(
            {
                "action": "run_third_htr_provider",
                "failure_domain": "stage5_recognition",
                "reason": (
                    "Use a materially different HTR provider to obtain new "
                    "independent visual evidence."
                ),
            }
        )

    if alternate_visual_htr_available:
        candidate_actions.append(
            {
                "action": "run_alternate_visual_htr",
                "failure_domain": "stage5_recognition_input",
                "reason": (
                    "Run the existing HTR path on a genuinely different visual "
                    "evidence view such as RAW versus BALANCED."
                ),
            }
        )

    if expanded_context_retry_available:
        candidate_actions.append(
            {
                "action": "run_expanded_context_retrieval",
                "failure_domain": "stage6_context",
                "reason": (
                    "Expand the attested corpus or retrieval channel and rerun "
                    "the evidence-constrained semantic substages."
                ),
            }
        )

    unattempted_actions = [
        row
        for row in candidate_actions
        if row["action"] not in attempted_actions
    ]

    status = "ok"
    decision = "proceed_to_translation_validation"
    next_action = "proceed_to_translation_validation"
    failure_domain = None
    scholar_review_required = False
    machine_retry_exhausted = False
    selected_retry_action = None
    exhaustion_reason = None

    if S < policy.stage4_min_segmentation_readiness:
        status = "blocked"
        decision = "retry_stage4_or_upstream"
        next_action = "retry_stage4_or_upstream"
        failure_domain = "stage4_segmentation"

        complaints.append("LOW_STAGE4_SEGMENTATION_READINESS")
        recommendations.append(
            "Use an automatic Stage-4/upstream retry before spending more "
            "recognition or semantic compute."
        )

    elif (
        T >= policy.translation_trust_threshold
        and normalized_complete
        and unresolved_lines == 0
    ):
        status = "ok"
        decision = "proceed_to_translation_validation"
        next_action = "proceed_to_translation_validation"

        recommendations.append(
            "Machine transcription trust is sufficient for translation "
            "validation. This still does not imply scholar-verified accuracy."
        )

    else:
        status = "retry_required"

        if H < policy.htr_low_readiness and S >= 0.75:
            failure_domain = "stage5_recognition"
            complaints.append(
                "HTR_READINESS_REMAINS_LOW_AFTER_STAGE6_RECOVERY"
            )
        elif unresolved_lines > 0:
            failure_domain = "stage6_reconstruction"
            complaints.append(
                "STAGE6_RECONSTRUCTION_REMAINS_UNRESOLVED"
            )
        else:
            failure_domain = "stage6_semantic_trust"
            complaints.append(
                "SEMANTIC_TRANSCRIPTION_TRUST_BELOW_THRESHOLD"
            )

        budget_exhausted = (
            retry_attempts >= policy.max_machine_retry_attempts
        )

        if not budget_exhausted and unattempted_actions:
            selected = unattempted_actions[0]
            selected_retry_action = selected["action"]
            decision = "machine_retry_required"
            next_action = selected["action"]

            recommendations.append(selected["reason"])
            recommendations.append(
                "After the retry, recompute H(p), rerun Stage 6A-6F as "
                "applicable, and compare ΔH and ΔT before any human escalation."
            )

        else:
            machine_retry_exhausted = True
            scholar_review_required = True
            status = "review_required"
            decision = "scholar_review_after_machine_exhaustion"
            next_action = "route_to_scholar_review"

            if budget_exhausted:
                exhaustion_reason = (
                    "configured_machine_retry_budget_exhausted"
                )
            else:
                exhaustion_reason = (
                    "no_additional_executable_machine_strategy_available"
                )

            recommendations.append(
                "Stage 6 semantic recovery has completed and no further "
                "executable machine retry remains under the current capability "
                "set. Scholar review is now permitted."
            )

            if not third_provider_available:
                recommendations.append(
                    "If a materially different Provider C is implemented later, "
                    "enable that capability and it will take precedence over "
                    "scholar escalation on future runs."
                )

    return {
        "layer": "layer6",
        "stage": "stage6_semantic_interpretation",
        "routing_version": STAGE6_ROUTING_VERSION,
        "status": status,
        "signals": {
            "S": round(S, 4),
            "H": round(H, 4),
            "T": round(T, 4),
            "interpretation": {
                "S": "Stage 4 segmentation readiness.",
                "H": (
                    "Stage 5 HTR system readiness; not recognition accuracy."
                ),
                "T": (
                    "Stage 6 semantic/transcription trust/readiness; "
                    "not calibrated probability, CER or WER."
                ),
            },
        },
        "stage6_completion": completion,
        "semantic_trust": {
            "page_T": round(T, 4),
            "unresolved_lines": unresolved_lines,
            "translation_eligible_lines": translation_eligible_lines,
            "normalized_page_complete": normalized_complete,
        },
        "retry_state": {
            "attempt_count": retry_attempts,
            "attempted_actions": attempted_actions,
            "max_machine_retry_attempts": policy.max_machine_retry_attempts,
        },
        "capabilities": {
            "third_htr_provider": third_provider_available,
            "alternate_visual_htr": alternate_visual_htr_available,
            "expanded_context_retrieval": expanded_context_retry_available,
        },
        "candidate_machine_actions": candidate_actions,
        "unattempted_machine_actions": unattempted_actions,
        "selected_retry_action": selected_retry_action,
        "failure_domain": failure_domain,
        "complaints": list(dict.fromkeys(complaints)),
        "recommendations": recommendations,
        "validation_trace": validation_trace,
        "decision": decision,
        "next_action": next_action,
        "machine_retry_exhausted": machine_retry_exhausted,
        "machine_retry_exhaustion_reason": exhaustion_reason,
        "scholar_review_required": scholar_review_required,
        "scholar_review_policy": {
            "allowed_before_stage6_completion": False,
            "allowed_while_executable_machine_retry_remains": False,
            "allowed_after_machine_exhaustion": True,
        },
        "policy": asdict(policy),
        "audit_note": (
            "Stage 6G is the first orchestration point permitted to authorize "
            "scholar review. It does so only after Stage 6 trust evaluation and "
            "machine-retry exhaustion under the explicitly declared runtime "
            "capabilities."
        ),
    }
