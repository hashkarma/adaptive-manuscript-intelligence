from __future__ import annotations

from core.artifact_store import ArtifactStore
from orchestration.signals import (
    compute_C_signal,
    compute_B_signal,
    compute_G_signal,
    compute_L_signal,
    compute_S_signal,
    compute_R_signal,
    extract_H_signal,
    extract_T_signal,
)
from orchestration.rules import (
    route_decision,
    route_after_segmentation,
    route_after_htr,
    route_after_semantic_trust,
)


def save_layer_report(
    store: ArtifactStore,
    layer_name: str,
    metrics: dict,
    status: str,
    complaints: list[str],
    *,
    next_action: str | None = None,
    recommendations: list[str] | None = None,
    signals: dict | None = None,
) -> dict:
    report = {
        "layer": layer_name,
        "status": status,
        "metrics": metrics,
        "complaints": complaints,
    }

    if next_action is not None:
        report["next_action"] = next_action
    if recommendations is not None:
        report["recommendations"] = recommendations
    if signals is not None:
        report["signals"] = signals

    store.write_json(f"orchestration/{layer_name}_report.json", report)
    return report


def simple_layer_status(metrics: dict, complaints: list[str]) -> str:
    return "warning" if len(complaints) >= 2 else "ok"


def evaluate_and_save_layer1(store: ArtifactStore, metrics: dict) -> dict:
    complaints = []
    if float(metrics.get("contrast_gain", 0.0)) < 5:
        complaints.append("Readability improvement is limited.")
    if float(metrics.get("binary_foreground_ratio", 0.0)) < 0.01:
        complaints.append("Binary result appears too sparse.")
    status = simple_layer_status(metrics, complaints)
    return save_layer_report(store, "layer1", metrics, status, complaints)


def evaluate_and_save_layer2(store: ArtifactStore, metrics: dict) -> dict:
    complaints = []
    if float(metrics.get("damage_ratio", 0.0)) > 0.30:
        complaints.append("A large portion of the page appears damaged.")
    if float(metrics.get("uncertainty_ratio", 0.0)) > 0.50:
        complaints.append(
            "A large portion of the page has low readability confidence."
        )
    status = simple_layer_status(metrics, complaints)
    return save_layer_report(store, "layer2", metrics, status, complaints)


def evaluate_and_save_layer3(store: ArtifactStore, metrics: dict) -> dict:
    complaints = []
    if int(metrics.get("num_regions", 0)) == 0:
        complaints.append("No text regions were detected.")
    if float(metrics.get("text_coverage_ratio", 0.0)) < 0.03:
        complaints.append("Detected text coverage is very low.")
    status = simple_layer_status(metrics, complaints)
    return save_layer_report(store, "layer3", metrics, status, complaints)


def evaluate_and_save_layer4(store: ArtifactStore, metrics: dict) -> dict:
    """
    Evaluate Research Stage 4.

    Stage 4 cannot route directly to scholar review. Poor evidence creates an
    automatic Stage-4/upstream retry. Human review is reserved for Stage 6G.
    """
    num_lines = int(metrics.get("num_lines", 0))
    losslessness = float(metrics.get("losslessness_score", 0.0))
    orphan_ratio = float(metrics.get("orphan_ink_ratio", 1.0))
    duplicate_ratio = float(metrics.get("duplicate_assignment_ratio", 1.0))
    consistency = float(metrics.get("line_height_consistency", 0.0))
    merged_regions = int(metrics.get("suspected_merged_regions", 0))
    segmentation_confidence = float(
        metrics.get("segmentation_confidence", 0.0)
    )
    boundary_uncertainty = float(
        metrics.get("mean_boundary_uncertainty", 1.0)
    )

    complaints: list[str] = []
    recommendations: list[str] = []
    hard_stop = False
    retry_needed = False

    if num_lines <= 0:
        hard_stop = True
        complaints.append("No manuscript text lines were detected.")
        recommendations.append(
            "Retry Research Stage 3 layout analysis and Stage 4 segmentation "
            "with an alternate configuration."
        )

    if losslessness < 0.90:
        hard_stop = True
        complaints.append(
            "Too much foreground ink was lost during line segmentation."
        )
        recommendations.append(
            "Do not continue to HTR; retry segmentation with safer boundaries."
        )
    elif losslessness < 0.97:
        retry_needed = True
        complaints.append(
            "Ink preservation is below the preferred Stage 4 threshold."
        )
        recommendations.append(
            "Retry Stage 4 with an alternate boundary/peak configuration."
        )

    if segmentation_confidence < 0.55:
        hard_stop = True
        complaints.append(
            "Overall segmentation confidence is too low for reliable HTR input."
        )
        recommendations.append(
            "Retry Stage 4 after verifying upstream layout output."
        )
    elif segmentation_confidence < 0.75:
        retry_needed = True
        complaints.append("Segmentation confidence is only moderate.")
        recommendations.append(
            "Retry Stage 4 and compare the alternative segmentation trace."
        )

    if orphan_ratio > 0.03:
        retry_needed = True
        complaints.append(
            "Too much manuscript ink remains unassigned to detected lines."
        )
        recommendations.append(
            "Retry Stage 4 and inspect the orphan-ink mask."
        )

    if duplicate_ratio > 0.01:
        retry_needed = True
        complaints.append(
            "Some foreground ink is assigned to more than one line crop."
        )
        recommendations.append(
            "Retry Stage 4 with non-overlapping line boundaries."
        )

    if consistency < 0.70:
        retry_needed = True
        complaints.append(
            "Detected line heights are inconsistent; merged or fragmented "
            "lines may remain."
        )
        recommendations.append(
            "Retry Stage 4 using alternate line-peak/valley parameters."
        )

    if merged_regions > 0:
        retry_needed = True
        complaints.append(
            f"{merged_regions} suspected merged line region(s) remain."
        )
        recommendations.append(
            "Retry Stage 4 before sending line crops to HTR."
        )

    if boundary_uncertainty > 0.30:
        retry_needed = True
        complaints.append("Average line-boundary uncertainty is high.")
        recommendations.append(
            "Inspect the uncertainty map and retry Stage 4 if boundaries "
            "cross glyph modifiers."
        )

    S = compute_S_signal(metrics)

    if hard_stop:
        status = "poor"
        next_action = "retry_stage4_or_upstream"
    elif retry_needed:
        status = "warning"
        next_action = "retry_stage4"
    else:
        status = "ok"
        next_action = "proceed_to_htr"

    report = save_layer_report(
        store,
        "layer4",
        metrics,
        status,
        complaints,
        next_action=next_action,
        recommendations=recommendations,
        signals={
            "S": S,
            "interpretation": "Segmentation readiness for downstream HTR",
        },
    )

    report["scholar_review_required"] = False
    report["scholar_review_deferred_until_stage6"] = True
    store.write_json("orchestration/layer4_report.json", report)
    return report


def evaluate_and_save_layer5(
    store: ArtifactStore,
    stage5_readiness: dict,
    *,
    third_provider_available: bool = False,
) -> dict:
    """
    Evaluate Stage 5 using the frozen Stage 5F readiness artifact.

    Low H(p) with healthy S(p) routes into Stage 6 deep reconstruction.
    """
    H = extract_H_signal(stage5_readiness)

    routing = route_after_htr(
        stage5_readiness,
        third_provider_available=third_provider_available,
    )

    page = stage5_readiness.get("page", {})
    evidence = page.get("evidence", {})

    metrics = {
        "htr_readiness_H": H,
        "completion": float(evidence.get("completion", 0.0)),
        "script_integrity": float(evidence.get("script_integrity", 0.0)),
        "decoder_reliability": float(
            evidence.get("decoder_reliability", 0.0)
        ),
        "sequence_quality": float(evidence.get("sequence_quality", 0.0)),
        "cross_provider_agreement": float(
            evidence.get("cross_provider_agreement", 0.0)
        ),
        "total_lines": int(page.get("total_lines", 0)),
        "low_readiness_lines": int(page.get("low_readiness_lines", 0)),
        "low_agreement_lines": int(page.get("low_agreement_lines", 0)),
        "ground_truth_available": bool(
            page.get("ground_truth_available", False)
        ),
        "cer": page.get("cer"),
        "wer": page.get("wer"),
        "recognition_accuracy_calibrated": bool(
            page.get("recognition_accuracy_calibrated", False)
        ),
    }

    report = {
        "layer": "layer5",
        "status": routing["status"],
        "metrics": metrics,
        "complaints": routing.get("complaints", []),
        "recommendations": routing.get("recommendations", []),
        "signals": routing["signals"],
        "evidence": routing["evidence"],
        "decision": routing["decision"],
        "next_action": routing["next_action"],
        "stage6_reconstruction_mode": routing.get(
            "stage6_reconstruction_mode"
        ),
        "hard_gates": routing.get("hard_gates", []),
        "routing_version": routing.get("routing_version"),
        "source_readiness_version": routing.get("source_readiness_version"),
        "third_provider_available": routing.get(
            "third_provider_available", False
        ),
        "third_provider_reserved_for_post_stage6_retry": routing.get(
            "third_provider_reserved_for_post_stage6_retry", False
        ),
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_adaptive_retries_exhausted": True,
        "ground_truth": routing.get("ground_truth", {}),
        "policy": routing.get("policy", {}),
        "audit_note": routing.get("audit_note"),
    }

    store.write_json("orchestration/layer5_report.json", report)
    return report


def evaluate_and_save_layer6(
    store: ArtifactStore,
    stage6_trust: dict,
    stage6_reconstruction: dict,
    stage5_readiness: dict,
    layer4_report: dict,
    *,
    third_provider_available: bool = False,
    alternate_visual_htr_available: bool = False,
    expanded_context_retry_available: bool = False,
    retry_state: dict | None = None,
) -> dict:
    """
    Evaluate Stage 6F T(p) and execute Stage 6G adaptive routing.

    This is the first evaluator entry point allowed to produce
    scholar_review_required=True.
    """
    T = extract_T_signal(stage6_trust)

    routing = route_after_semantic_trust(
        stage6_trust,
        stage6_reconstruction,
        stage5_readiness,
        layer4_report,
        third_provider_available=third_provider_available,
        alternate_visual_htr_available=alternate_visual_htr_available,
        expanded_context_retry_available=expanded_context_retry_available,
        retry_state=retry_state,
    )

    trust_metrics = stage6_trust.get("metrics", {}) or {}
    reconstruction_metrics = (
        stage6_reconstruction.get("metrics", {}) or {}
    )

    metrics = {
        "semantic_transcription_trust_T": T,
        "page_T_mean": trust_metrics.get("page_T_mean"),
        "page_T_status": trust_metrics.get("page_T_status"),
        "unresolved_lines": trust_metrics.get("unresolved_lines"),
        "adaptive_retry_required_lines": trust_metrics.get(
            "adaptive_retry_required_lines"
        ),
        "translation_eligible_lines": trust_metrics.get(
            "translation_eligible_lines"
        ),
        "reconstructed_lines": reconstruction_metrics.get(
            "reconstructed_lines"
        ),
        "partially_supported_lines": reconstruction_metrics.get(
            "partially_supported_lines"
        ),
        "abstained_lines": reconstruction_metrics.get("abstained_lines"),
        "normalized_lines": reconstruction_metrics.get("normalized_lines"),
        "T_is_calibrated_probability": False,
        "T_is_accuracy": False,
        "T_is_CER_or_WER": False,
    }

    report = {
        "layer": "layer6",
        "status": routing["status"],
        "metrics": metrics,
        "complaints": routing.get("complaints", []),
        "recommendations": routing.get("recommendations", []),
        "signals": routing["signals"],
        "decision": routing["decision"],
        "next_action": routing["next_action"],
        "failure_domain": routing.get("failure_domain"),
        "selected_retry_action": routing.get("selected_retry_action"),
        "machine_retry_exhausted": routing.get(
            "machine_retry_exhausted", False
        ),
        "machine_retry_exhaustion_reason": routing.get(
            "machine_retry_exhaustion_reason"
        ),
        "scholar_review_required": routing.get(
            "scholar_review_required", False
        ),
        "stage6_completion": routing.get("stage6_completion", {}),
        "semantic_trust": routing.get("semantic_trust", {}),
        "retry_state": routing.get("retry_state", {}),
        "capabilities": routing.get("capabilities", {}),
        "candidate_machine_actions": routing.get(
            "candidate_machine_actions", []
        ),
        "unattempted_machine_actions": routing.get(
            "unattempted_machine_actions", []
        ),
        "validation_trace": routing.get("validation_trace", []),
        "scholar_review_policy": routing.get(
            "scholar_review_policy", {}
        ),
        "policy": routing.get("policy", {}),
        "routing_version": routing.get("routing_version"),
        "audit_note": routing.get("audit_note"),
    }

    store.write_json("orchestration/layer6_report.json", report)
    return report


def _stage5_summary(layer5_report: dict) -> str:
    decision = str(layer5_report.get("decision", ""))

    summaries = {
        "proceed_to_stage6_deep_reconstruction": (
            "Stage 5 HTR evidence is weak while segmentation is usable. "
            "Proceed to Stage 6 deep, evidence-constrained reconstruction."
        ),
        "proceed_to_stage6_enhanced_reconstruction": (
            "Stage 5 HTR evidence is incomplete or intermediate for unattended "
            "acceptance. Proceed to Stage 6 in enhanced reconstruction mode."
        ),
        "provisionally_proceed_to_stage6": (
            "Stage 5 HTR system readiness passed the provisional downstream "
            "gate. This is not scholar-verified transcription accuracy."
        ),
        "return_to_stage4": (
            "Stage 5 cannot continue because segmentation readiness is below "
            "the required gate."
        ),
        "retry_or_switch_htr_provider": (
            "Stage 5 provider execution is incomplete; retry or switch HTR "
            "provider before downstream processing."
        ),
    }

    return summaries.get(
        decision,
        "Stage 5 produced an adaptive HTR routing decision.",
    )


def _stage6_summary(layer6_report: dict) -> str:
    decision = str(layer6_report.get("decision", ""))

    summaries = {
        "proceed_to_translation_validation": (
            "Stage 6 produced sufficiently trusted, complete machine "
            "transcription evidence to proceed to translation validation."
        ),
        "machine_retry_required": (
            "Stage 6 trust remains too low; execute the selected machine retry "
            "and recompute H(p) and T(p) before human escalation."
        ),
        "retry_stage4_or_upstream": (
            "Stage 6 diagnosed an unresolved upstream segmentation problem; "
            "return to an automatic Stage-4/upstream retry."
        ),
        "scholar_review_after_machine_exhaustion": (
            "Stage 6 semantic recovery is complete and no executable machine "
            "retry remains under the current capability set. Scholar review is "
            "now permitted."
        ),
    }

    return summaries.get(
        decision,
        "Stage 6 produced an adaptive semantic/transcription routing decision.",
    )


def finalize_orchestration(store: ArtifactStore, layer_reports: dict) -> dict:
    """
    Final adaptive routing state.

    Phases:
      pre_stage4 -> post_stage4 -> post_stage5 -> post_stage6

    H(p) and T(p) are carried, never recomputed here.
    """
    layer1_metrics = layer_reports.get("layer1", {}).get("metrics", {})
    layer2_metrics = layer_reports.get("layer2", {}).get("metrics", {})
    layer3_metrics = layer_reports.get("layer3", {}).get("metrics", {})
    layer4_report = layer_reports.get("layer4", {}) or {}
    layer4_metrics = layer4_report.get("metrics", {})
    layer5_report = layer_reports.get("layer5", {}) or {}
    layer6_report = layer_reports.get("layer6", {}) or {}

    C = compute_C_signal(layer1_metrics)
    B = compute_B_signal(layer1_metrics)
    G = compute_G_signal(layer2_metrics)
    L = compute_L_signal(layer3_metrics)
    R = compute_R_signal(C, B, G, L)

    signals = {
        "C": C,
        "B": B,
        "G": G,
        "L": L,
        "R": R,
    }

    interpretations = {
        "C": "Enhancement adequacy — Research Stage 1 evidence",
        "B": "Estimated bleed-through severity — provisional proxy",
        "G": "Upstream geometry / structural reliability",
        "L": "Layout readiness — Research Stage 3 evidence",
        "R": "Pre-segmentation routing readiness across Research Stages 1-3",
    }

    S = None
    H = None

    if layer4_metrics:
        S = layer4_report.get("signals", {}).get("S")
        if S is None:
            S = compute_S_signal(layer4_metrics)
        S = float(S)
        signals["S"] = S
        interpretations["S"] = (
            "Segmentation readiness — Research Stage 4 validation evidence"
        )

    if layer5_report:
        layer5_signals = layer5_report.get("signals", {}) or {}
        H = layer5_signals.get("H")
        if H is None:
            H = layer5_report.get("metrics", {}).get("htr_readiness_H")

        if H is None:
            raise ValueError(
                "Layer 5 report exists but does not contain H(p)."
            )

        H = float(H)
        signals["H"] = H
        interpretations["H"] = (
            "HTR system readiness — Research Stage 5 evidence; "
            "not recognition accuracy"
        )

        if S is None:
            stage5_S = layer5_signals.get("S")
            if stage5_S is None:
                raise ValueError(
                    "Layer 5 report exists but neither Layer 4 nor Stage 5 "
                    "contains S(p)."
                )
            S = float(stage5_S)
            signals["S"] = S

    if layer6_report:
        layer6_signals = layer6_report.get("signals", {}) or {}
        T = layer6_signals.get("T")
        if T is None:
            T = layer6_report.get("metrics", {}).get(
                "semantic_transcription_trust_T"
            )

        if T is None:
            raise ValueError(
                "Layer 6 report exists but does not contain T(p)."
            )

        T = float(T)
        signals["T"] = T
        interpretations["T"] = (
            "Semantic/transcription trust — Research Stage 6F evidence; "
            "not calibrated probability, CER, WER or accuracy"
        )

        if H is None:
            H = float(layer6_signals["H"])
            signals["H"] = H

        if S is None:
            S = float(layer6_signals["S"])
            signals["S"] = S

        policy = layer6_report.get("policy", {}) or {}

        decision = {
            "overall_status": layer6_report.get(
                "status", "review_required"
            ),
            "next_action": layer6_report.get(
                "next_action", "route_to_scholar_review"
            ),
            "summary": _stage6_summary(layer6_report),
            "complaints": layer6_report.get("complaints", []),
            "recommendations": layer6_report.get("recommendations", []),
            "advisories": [],
            "validation_trace": layer6_report.get(
                "validation_trace", []
            ),
            "critical_signal_checks": {
                "S_htr_gate_met": float(S) >= 0.75,
                "H_intermediate_readiness_met": float(H) >= 0.50,
                "T_translation_gate_met": float(T) >= float(
                    policy.get("translation_trust_threshold", 0.65)
                ),
            },
            "semantic_decision": layer6_report.get("decision"),
            "selected_retry_action": layer6_report.get(
                "selected_retry_action"
            ),
            "failure_domain": layer6_report.get("failure_domain"),
            "machine_retry_exhausted": layer6_report.get(
                "machine_retry_exhausted", False
            ),
            "machine_retry_exhaustion_reason": layer6_report.get(
                "machine_retry_exhaustion_reason"
            ),
            "scholar_review_required": layer6_report.get(
                "scholar_review_required", False
            ),
            "stage6_completion": layer6_report.get(
                "stage6_completion", {}
            ),
            "retry_state": layer6_report.get("retry_state", {}),
            "capabilities": layer6_report.get("capabilities", {}),
            "scholar_review_policy": layer6_report.get(
                "scholar_review_policy", {}
            ),
        }
        phase = "post_stage6"

    elif layer5_report:
        layer5_signals = layer5_report.get("signals", {}) or {}
        stage5_S = layer5_signals.get("S")
        s_consistent = True

        if stage5_S is not None:
            s_consistent = abs(float(stage5_S) - float(S)) <= 0.01

        evidence = layer5_report.get("evidence", {}) or {}
        policy = layer5_report.get("policy", {}) or {}

        decoder_accept = float(
            evidence.get("decoder_reliability", 0.0)
        ) >= float(
            policy.get("min_decoder_reliability_for_accept", 0.55)
        )

        agreement_accept = float(
            evidence.get("cross_provider_agreement", 0.0)
        ) >= float(
            policy.get("min_cross_provider_agreement_for_accept", 0.75)
        )

        advisories: list[str] = []
        hard_gates = list(layer5_report.get("hard_gates", []))

        if not s_consistent:
            advisories.append(
                "Stage 4 S(p) and the S(p) stored in the Stage 5 readiness "
                "artifact differ by more than 0.01; verify artifact freshness."
            )
            hard_gates.append("STAGE4_STAGE5_S_SIGNAL_MISMATCH")

        decision = {
            "overall_status": layer5_report.get("status", "warning"),
            "next_action": layer5_report.get(
                "next_action",
                "proceed_to_stage6_deep_reconstruction",
            ),
            "summary": _stage5_summary(layer5_report),
            "complaints": layer5_report.get("complaints", []),
            "recommendations": layer5_report.get("recommendations", []),
            "advisories": advisories,
            "validation_trace": [
                (
                    "Stage 4 segmentation readiness is strong enough to "
                    "localize the current uncertainty to Research Stage 5 HTR."
                    if float(S) >= 0.75
                    else
                    "Stage 4 segmentation readiness remains below the normal "
                    "HTR progression gate."
                ),
                (
                    f"Stage 5 produced H(p)={H:.4f}; H(p) is system readiness "
                    "and must not be interpreted as transcription accuracy."
                ),
                (
                    "Scholar review is intentionally deferred until Stage 6 "
                    "semantic recovery and adaptive retry evaluation."
                ),
            ],
            "critical_signal_checks": {
                "S_htr_gate_met": float(S) >= 0.75,
                "stage4_stage5_S_consistent": s_consistent,
                "H_high_readiness_met": H >= float(
                    policy.get("htr_high_readiness", 0.75)
                ),
                "decoder_accept_gate_met": decoder_accept,
                "cross_provider_agreement_accept_gate_met": (
                    agreement_accept
                ),
            },
            "htr_decision": layer5_report.get("decision"),
            "hard_gates": list(dict.fromkeys(hard_gates)),
            "htr_evidence": evidence,
            "ground_truth": layer5_report.get("ground_truth", {}),
            "scholar_review_required": False,
            "scholar_review_deferred_until_stage6": True,
        }
        phase = "post_stage5"

    elif layer4_metrics:
        decision = route_after_segmentation(
            signals,
            str(layer4_report.get("status", "warning")),
        )
        phase = "post_stage4"

    else:
        decision = route_decision(signals)
        phase = "pre_stage4"

    final = {
        "phase": phase,
        "signals": signals,
        "signal_interpretation": interpretations,
        **decision,
    }

    if phase == "pre_stage4":
        store.write_json(
            "orchestration/pre_stage4_decision.json", final
        )

    if phase == "post_stage5":
        store.write_json(
            "orchestration/post_stage5_decision.json", final
        )

    if phase == "post_stage6":
        store.write_json(
            "orchestration/post_stage6_decision.json", final
        )

    store.write_json("orchestration/final_decision.json", final)
    return final
