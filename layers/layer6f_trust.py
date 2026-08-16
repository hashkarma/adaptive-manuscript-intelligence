from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from core.artifact_store import ArtifactStore


LAYER6F_VERSION = "0.1.0-semantic-transcription-trust"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6F input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _supported_span_coverage(line: Dict[str, Any]) -> float:
    spans = line.get("supported_spans", [])
    if not isinstance(spans, list) or not spans:
        return 0.0

    positions: List[float] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        positions.append(_clamp01(_safe_float(span.get("position"), 0.5)))

    if not positions:
        return 0.0
    if len(positions) == 1:
        return 0.12

    spread = max(positions) - min(positions)
    count_factor = min(1.0, len(positions) / 5.0)
    return _clamp01(0.70 * spread + 0.30 * count_factor)


def _reconstruction_component(line: Dict[str, Any]) -> float:
    decision = str(line.get("machine_decision", "") or "")
    normalized = line.get("normalized_devanagari")
    spans = line.get("supported_spans", [])

    if decision == "reconstruct_supported_line" and normalized:
        return 1.0
    if (
        decision == "partial_reconstruction_with_abstention"
        and isinstance(spans, list)
        and spans
    ):
        return 0.45
    if decision == "abstain_insufficient_evidence":
        return 0.0
    return 0.10


def _context_component(line: Dict[str, Any]) -> float:
    raw = _clamp01(_safe_float(line.get("best_context_score"), 0.0))
    return min(0.70, raw)


def _weighted_geometric_mean(
    components: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    for name, weight in weights.items():
        if weight > 0 and _clamp01(components.get(name, 0.0)) <= 0.0:
            return 0.0

    log_sum = 0.0
    for name, weight in weights.items():
        if weight <= 0:
            continue
        value = _clamp01(components.get(name, 0.0))
        log_sum += (weight / total_weight) * math.log(value)

    return _clamp01(math.exp(log_sum))


def _trust_status(T: float, *, has_normalized_text: bool, decision: str) -> str:
    if decision == "abstain_insufficient_evidence":
        return "untrusted_unresolved"
    if not has_normalized_text:
        return "partial_evidence_not_final" if T >= 0.55 else "low_trust_partial"
    if T >= 0.80:
        return "high_machine_trust"
    if T >= 0.65:
        return "moderate_machine_trust"
    if T >= 0.45:
        return "low_machine_trust"
    return "untrusted_reconstruction"


def _recommended_next_action(line: Dict[str, Any], *, T: float) -> str:
    decision = str(line.get("machine_decision", "") or "")
    if decision == "abstain_insufficient_evidence" or T < 0.45:
        return "adaptive_retry_required"
    if line.get("normalized_devanagari") is None:
        return "adaptive_retry_required"
    if T < 0.65:
        return "adaptive_retry_recommended"
    return "eligible_for_translation_validation"


def _line_trust(line: Dict[str, Any]) -> Dict[str, Any]:
    H = _clamp01(_safe_float(line.get("H"), 0.0))
    agreement = _clamp01(_safe_float(line.get("cross_provider_agreement"), 0.0))
    stage6b_candidate = _clamp01(
        _safe_float(line.get("best_stage6b_candidate_score"), 0.0)
    )
    visual_morph = _clamp01(
        _safe_float(line.get("visually_supported_candidate_ratio"), 0.0)
    )
    span_coverage = _supported_span_coverage(line)
    reconstruction = _reconstruction_component(line)
    context = _context_component(line)

    components = {
        "htr_readiness": H,
        "cross_provider_agreement": agreement,
        "stage6b_candidate_evidence": stage6b_candidate,
        "visual_morphological_support": visual_morph,
        "supported_span_coverage": span_coverage,
        "reconstruction_completion": reconstruction,
        "contextual_support_capped": context,
    }

    weights = {
        "htr_readiness": 0.22,
        "cross_provider_agreement": 0.18,
        "stage6b_candidate_evidence": 0.12,
        "visual_morphological_support": 0.20,
        "supported_span_coverage": 0.12,
        "reconstruction_completion": 0.11,
        "contextual_support_capped": 0.05,
    }

    T = _weighted_geometric_mean(components, weights)
    decision = str(line.get("machine_decision", "") or "")
    has_normalized_text = bool(line.get("normalized_devanagari"))

    if (
        decision == "abstain_insufficient_evidence"
        or visual_morph <= 0.0
        or reconstruction <= 0.0
    ):
        T = 0.0

    status = _trust_status(
        T,
        has_normalized_text=has_normalized_text,
        decision=decision,
    )

    return {
        "line_id": line.get("line_id"),
        "reading_order": line.get("reading_order"),
        "T": round(T, 4),
        "T_status": status,
        "T_is_calibrated_probability": False,
        "T_is_accuracy": False,
        "T_is_CER_or_WER": False,
        "components": {key: round(value, 4) for key, value in components.items()},
        "weights": weights,
        "machine_decision_from_stage6e": decision,
        "normalized_devanagari_available": has_normalized_text,
        "recommended_next_action": _recommended_next_action(line, T=T),
        "abstention_reasons": line.get("abstention_reasons", []),
    }


def run_layer6f_semantic_transcription_trust(store: ArtifactStore) -> Dict[str, Any]:
    """
    Stage 6F — semantic/transcription trust T(p).

    T(p) is an uncalibrated evidence-fusion trust/readiness score, not a
    probability of correctness and not CER/WER/accuracy. A line that Stage 6E
    explicitly abstains from receives T=0 because there is no promoted
    transcription to trust.
    """
    run_dir = Path(store.run_dir)
    stage6e = _load_json(run_dir / "L6" / "stage6e_manifest.json")

    lines = stage6e.get("lines")
    if not isinstance(lines, list) or not lines:
        raise RuntimeError("Stage 6F requires Stage 6E line results.")

    line_results = [
        _line_trust(line)
        for line in sorted(
            lines,
            key=lambda row: int(row.get("reading_order", 10**9)),
        )
    ]

    line_T = [float(row["T"]) for row in line_results]
    page_T_mean = sum(line_T) / float(len(line_T))
    page_T_min = min(line_T)

    unresolved_lines = sum(
        row["T_status"] == "untrusted_unresolved" for row in line_results
    )
    retry_required_lines = sum(
        row["recommended_next_action"] == "adaptive_retry_required"
        for row in line_results
    )
    translation_eligible_lines = sum(
        row["recommended_next_action"] == "eligible_for_translation_validation"
        for row in line_results
    )

    page_T = round(page_T_min, 4)
    if page_T >= 0.80:
        page_status = "high_machine_trust"
    elif page_T >= 0.65:
        page_status = "moderate_machine_trust"
    elif page_T >= 0.45:
        page_status = "low_machine_trust"
    else:
        page_status = "untrusted_or_unresolved"

    metrics = {
        "algorithm_version": LAYER6F_VERSION,
        "stage6_substage": "6F_semantic_transcription_trust",
        "num_lines": len(line_results),
        "page_T": page_T,
        "page_T_mean": round(page_T_mean, 4),
        "page_T_conservative_policy": "minimum_line_T_for_complete_page_transcription",
        "page_T_status": page_status,
        "unresolved_lines": unresolved_lines,
        "adaptive_retry_required_lines": retry_required_lines,
        "translation_eligible_lines": translation_eligible_lines,
        "T_is_calibrated_probability": False,
        "T_is_accuracy": False,
        "T_is_CER_or_WER": False,
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_adaptive_retries_exhausted": True,
    }

    manifest = {
        "stage": "stage6_semantic_interpretation",
        "substage": "6F_semantic_transcription_trust",
        "version": LAYER6F_VERSION,
        "run_id": store.run_id,
        "status": "semantic_transcription_trust_computed",
        "T_definition": {
            "meaning": (
                "uncalibrated evidence-fusion score representing trust/readiness "
                "of the machine-produced semantic transcription"
            ),
            "not_probability": True,
            "not_accuracy": True,
            "not_CER": True,
            "not_WER": True,
            "explicit_abstention_maps_to_zero_trust": True,
        },
        "metrics": metrics,
        "lines": line_results,
        "next_action": "run_stage6g_adaptive_retry_controller",
        "human_review_policy": {
            "stage6f_low_T_is_not_immediate_scholar_stop": True,
            "scholar_review_deferred_until_adaptive_retry_budget_exhausted": True,
        },
        "audit_note": (
            "Stage 6F computes provisional T(p) using geometric evidence fusion. "
            "Missing visual/morphological support or Stage-6E abstention cannot "
            "be hidden by strong lexical/context signals. T(p) is not a calibrated "
            "probability or recognition accuracy metric."
        ),
    }

    store.write_json(
        "L6/transcription_trust.json",
        {
            "version": LAYER6F_VERSION,
            "run_id": store.run_id,
            "page_T": page_T,
            "page_T_mean": round(page_T_mean, 4),
            "page_T_status": page_status,
            "lines": line_results,
        },
    )
    store.write_json("L6/stage6f_manifest.json", manifest)

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
