from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.artifact_store import ArtifactStore


LAYER6E_VERSION = "0.1.0-abstention-capable-evidence-constrained-reconstruction"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6E input: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _line_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = payload.get("lines")

    if not isinstance(rows, list):
        rows = payload.get("rows")

    if not isinstance(rows, list):
        return {}

    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        line_id = str(row.get("line_id", "") or "")

        if line_id:
            result[line_id] = row

    return result


def _best_context_score(
    d1_line: Dict[str, Any],
    d2_line: Dict[str, Any],
) -> Tuple[float, str]:
    d1 = _safe_float(
        d1_line.get("best_context_evidence_score"),
        0.0,
    )

    d2 = _safe_float(
        d2_line.get("best_noisy_surface_score"),
        0.0,
    )

    if d2 >= d1:
        return d2, "stage6d2_noisy_surface_retrieval"

    return d1, "stage6d1_deterministic_context_retrieval"


def _candidate_position(
    cluster: Dict[str, Any],
) -> float:
    return _safe_float(
        cluster.get("mean_normalized_position"),
        0.5,
    )


def _eligible_supported_spans(
    stage6b_line: Dict[str, Any],
    stage6c_line: Dict[str, Any],
    *,
    H: float,
    agreement: float,
) -> List[Dict[str, Any]]:
    """
    A Stage-6E supported span must already have independent visual evidence
    AND morphological support.

    Retrieval is not allowed to create a supported span because retrieved text
    can be contextually related without being the manuscript text.
    """
    cluster_by_id = {
        str(cluster.get("cluster_id")): cluster
        for cluster in stage6b_line.get("candidate_clusters", [])
        if isinstance(cluster, dict)
    }

    spans: List[Dict[str, Any]] = []

    for validated in stage6c_line.get(
        "validated_candidate_clusters",
        [],
    ):
        if not isinstance(validated, dict):
            continue

        if (
            validated.get("stage6c_status")
            != "visually_and_morphologically_supported"
        ):
            continue

        cluster_id = str(validated.get("cluster_id", "") or "")
        cluster = cluster_by_id.get(cluster_id, {})

        if not bool(cluster.get("cross_provider_support", False)):
            continue

        stage6b_score = _safe_float(
            cluster.get("evidence_score"),
            0.0,
        )

        # Conservative minimums. These are engineering gates, not calibrated
        # probabilities.
        if H < 0.50 or agreement < 0.50 or stage6b_score < 0.50:
            continue

        token = str(
            cluster.get(
                "representative_devanagari",
                "",
            )
            or ""
        )

        if not token:
            continue

        spans.append(
            {
                "cluster_id": cluster_id,
                "token": token,
                "position": round(
                    _candidate_position(cluster),
                    4,
                ),
                "stage6b_evidence_score": round(
                    stage6b_score,
                    4,
                ),
                "stage6c_status": validated.get(
                    "stage6c_status"
                ),
                "cross_provider_support": True,
                "source_policy": (
                    "observed_HRT_token_only"
                ),
                "promotion_basis": (
                    "cross_provider_visual_support_plus_morphology"
                ),
            }
        )

    spans.sort(
        key=lambda row: row["position"]
    )

    return spans


def _diplomatic_from_supported_spans(
    spans: Sequence[Dict[str, Any]],
) -> str:
    if not spans:
        return "⟦unresolved⟧"

    parts: List[str] = ["⟦…⟧"]

    for span in spans:
        parts.append(str(span["token"]))
        parts.append("⟦…⟧")

    return " ".join(parts)


def _line_complete_enough(
    spans: Sequence[Dict[str, Any]],
    *,
    H: float,
    agreement: float,
) -> bool:
    """
    Stage 6E may only emit normalized line text if evidence is broad enough to
    cover the line rather than a few isolated tokens.
    """
    if H < 0.70 or agreement < 0.65:
        return False

    if len(spans) < 3:
        return False

    positions = [
        _safe_float(span.get("position"), 0.5)
        for span in spans
    ]

    if not positions:
        return False

    coverage_span = max(positions) - min(positions)

    return coverage_span >= 0.60


def _reconstruction_evidence_index(
    *,
    H: float,
    agreement: float,
    best_stage6b_candidate: float,
    visually_supported_ratio: float,
    best_context: float,
) -> float:
    """
    Engineering evidence index for Stage 6E orchestration.

    This is NOT T(p), confidence, probability, CER, or accuracy.
    """
    score = (
        0.30 * H
        + 0.25 * agreement
        + 0.20 * best_stage6b_candidate
        + 0.15 * visually_supported_ratio
        + 0.10 * best_context
    )

    return round(
        max(0.0, min(1.0, score)),
        4,
    )


def run_layer6e_evidence_constrained_reconstruction(
    store: ArtifactStore,
) -> Dict[str, Any]:
    """
    Stage 6E — abstention-capable evidence-constrained reconstruction.

    Inputs:
      L6/htr_input_table.json
      L6/stage6b_manifest.json
      L6/stage6c_manifest.json
      L6/stage6d_manifest.json
      L6/stage6d2_manifest.json

    Core guarantees:
      - no free Sanskrit generation;
      - retrieval cannot directly become manuscript text;
      - only observed HTR tokens with cross-provider + morphology support may
        become supported spans;
      - unresolved lines explicitly abstain;
      - normalized transcription is emitted only with broad evidence coverage;
      - T(p) remains unset until Stage 6F.
    """
    run_dir = Path(store.run_dir)

    stage6a = _load_json(
        run_dir / "L6" / "htr_input_table.json"
    )
    stage6b = _load_json(
        run_dir / "L6" / "stage6b_manifest.json"
    )
    stage6c = _load_json(
        run_dir / "L6" / "stage6c_manifest.json"
    )
    stage6d1 = _load_json(
        run_dir / "L6" / "stage6d_manifest.json"
    )
    stage6d2 = _load_json(
        run_dir / "L6" / "stage6d2_manifest.json"
    )

    a_map = _line_map(stage6a)
    b_map = _line_map(stage6b)
    c_map = _line_map(stage6c)
    d1_map = _line_map(stage6d1)
    d2_map = _line_map(stage6d2)

    line_ids = sorted(
        set(a_map)
        | set(b_map)
        | set(c_map)
        | set(d1_map)
        | set(d2_map),
        key=lambda line_id: int(
            a_map.get(line_id, {}).get(
                "reading_order",
                b_map.get(line_id, {}).get(
                    "reading_order",
                    10**9,
                ),
            )
        ),
    )

    if not line_ids:
        raise RuntimeError(
            "Stage 6E found no line-level evidence."
        )

    line_results: List[Dict[str, Any]] = []

    reconstructed_lines = 0
    partially_supported_lines = 0
    abstained_lines = 0
    normalized_lines = 0

    for line_id in line_ids:
        a = a_map.get(line_id, {})
        b = b_map.get(line_id, {})
        c = c_map.get(line_id, {})
        d1 = d1_map.get(line_id, {})
        d2 = d2_map.get(line_id, {})

        H = _safe_float(
            a.get(
                "htr_readiness_H",
                b.get(
                    "H",
                    c.get(
                        "H",
                        d1.get("H"),
                    ),
                ),
            ),
            0.0,
        )

        agreement = _safe_float(
            a.get(
                "cross_provider_agreement",
                b.get(
                    "cross_provider_agreement",
                    c.get(
                        "cross_provider_agreement",
                        d1.get(
                            "cross_provider_agreement"
                        ),
                    ),
                ),
            ),
            0.0,
        )

        clusters = b.get(
            "candidate_clusters",
            [],
        )

        best_stage6b_candidate = max(
            (
                _safe_float(
                    cluster.get(
                        "evidence_score"
                    ),
                    0.0,
                )
                for cluster in clusters
                if isinstance(cluster, dict)
            ),
            default=0.0,
        )

        validated = c.get(
            "validated_candidate_clusters",
            [],
        )

        visually_supported = sum(
            1
            for row in validated
            if isinstance(row, dict)
            and row.get("stage6c_status")
            == "visually_and_morphologically_supported"
        )

        visually_supported_ratio = (
            visually_supported
            / float(max(1, len(validated)))
        )

        best_context, context_source = (
            _best_context_score(
                d1,
                d2,
            )
        )

        supported_spans = (
            _eligible_supported_spans(
                b,
                c,
                H=H,
                agreement=agreement,
            )
        )

        line_complete = (
            _line_complete_enough(
                supported_spans,
                H=H,
                agreement=agreement,
            )
        )

        if line_complete:
            decision = (
                "reconstruct_supported_line"
            )
            reconstructed_lines += 1

        elif supported_spans:
            decision = (
                "partial_reconstruction_with_abstention"
            )
            partially_supported_lines += 1

        else:
            decision = (
                "abstain_insufficient_evidence"
            )
            abstained_lines += 1

        diplomatic = (
            _diplomatic_from_supported_spans(
                supported_spans
            )
        )

        normalized = None

        if line_complete:
            normalized = " ".join(
                span["token"]
                for span in supported_spans
            )
            normalized_lines += 1

        evidence_index = (
            _reconstruction_evidence_index(
                H=H,
                agreement=agreement,
                best_stage6b_candidate=best_stage6b_candidate,
                visually_supported_ratio=visually_supported_ratio,
                best_context=best_context,
            )
        )

        reasons: List[str] = []

        if H < 0.50:
            reasons.append(
                "LOW_HTR_READINESS"
            )

        if agreement < 0.50:
            reasons.append(
                "LOW_CROSS_PROVIDER_AGREEMENT"
            )

        if visually_supported == 0:
            reasons.append(
                "NO_VISUALLY_AND_MORPHOLOGICALLY_SUPPORTED_CANDIDATES"
            )

        if best_context < 0.62:
            reasons.append(
                "WEAK_CONTEXT_RETRIEVAL"
            )

        if not supported_spans:
            reasons.append(
                "NO_SPAN_ELIGIBLE_FOR_PROMOTION"
            )

        line_results.append(
            {
                "line_id": line_id,
                "reading_order": a.get(
                    "reading_order",
                    b.get(
                        "reading_order"
                    ),
                ),
                "H": round(H, 4),
                "cross_provider_agreement": round(
                    agreement,
                    4,
                ),
                "provider_a_top1_readable": a.get(
                    "provider_a_top1_readable"
                ),
                "provider_b_top1_readable": a.get(
                    "provider_b_top1_readable"
                ),
                "best_stage6b_candidate_score": round(
                    best_stage6b_candidate,
                    4,
                ),
                "visually_and_morphologically_supported_candidates": (
                    visually_supported
                ),
                "visually_supported_candidate_ratio": round(
                    visually_supported_ratio,
                    4,
                ),
                "best_context_score": round(
                    best_context,
                    4,
                ),
                "best_context_source": (
                    context_source
                ),
                "reconstruction_evidence_index": (
                    evidence_index
                ),
                "reconstruction_evidence_index_is_confidence": False,
                "supported_spans": supported_spans,
                "machine_decision": decision,
                "diplomatic_transcription": diplomatic,
                "normalized_devanagari": normalized,
                "translation_allowed": False,
                "abstention_reasons": reasons,
                "stage6e_status": (
                    "resolved"
                    if line_complete
                    else (
                        "partially_resolved"
                        if supported_spans
                        else "unresolved"
                    )
                ),
            }
        )

    mean_index = round(
        sum(
            row[
                "reconstruction_evidence_index"
            ]
            for row in line_results
        )
        / float(len(line_results)),
        4,
    )

    page_normalized = None

    if normalized_lines == len(line_results):
        page_normalized = "\n".join(
            str(
                row[
                    "normalized_devanagari"
                ]
            )
            for row in line_results
        )

    metrics = {
        "algorithm_version": LAYER6E_VERSION,
        "stage6_substage": (
            "6E_evidence_constrained_reconstruction"
        ),
        "num_lines": len(line_results),
        "reconstructed_lines": reconstructed_lines,
        "partially_supported_lines": (
            partially_supported_lines
        ),
        "abstained_lines": abstained_lines,
        "normalized_lines": normalized_lines,
        "page_reconstruction_evidence_index": (
            mean_index
        ),
        "page_reconstruction_evidence_index_is_confidence": False,
        "generated_unobserved_sanskrit_tokens": 0,
        "retrieval_promoted_directly_to_text": False,
        "final_diplomatic_transcription_available": True,
        "final_normalized_devanagari_available": (
            page_normalized is not None
        ),
        "final_semantic_trust_T": None,
        "translation_available": False,
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_exhausted": True,
    }

    manifest = {
        "stage": (
            "stage6_semantic_interpretation"
        ),
        "substage": (
            "6E_evidence_constrained_reconstruction"
        ),
        "version": LAYER6E_VERSION,
        "run_id": store.run_id,
        "status": (
            "reconstruction_evaluated_with_abstention"
        ),
        "safety_contract": {
            "free_sanskrit_generation_allowed": False,
            "retrieval_text_can_be_copied_directly": False,
            "only_observed_htr_tokens_may_form_supported_spans": True,
            "cross_provider_support_required_for_supported_spans": True,
            "morphological_support_required_for_supported_spans": True,
            "explicit_abstention_supported": True,
            "normalized_text_requires_broad_line_evidence": True,
            "T_computed_here": False,
            "translation_allowed": False,
        },
        "metrics": metrics,
        "page_normalized_devanagari": (
            page_normalized
        ),
        "lines": line_results,
        "next_action": (
            "run_stage6f_semantic_transcription_trust"
        ),
        "audit_note": (
            "Stage 6E is abstention-capable. It does not hallucinate missing "
            "Sanskrit. Retrieval evidence cannot become manuscript text. "
            "Supported spans must already be observed HTR evidence with "
            "cross-provider and morphological support. Unresolved material is "
            "explicitly marked as unresolved."
        ),
    }

    store.write_json(
        "L6/reconstruction_report.json",
        {
            "version": LAYER6E_VERSION,
            "run_id": store.run_id,
            "page_normalized_devanagari": (
                page_normalized
            ),
            "lines": line_results,
        },
    )

    store.write_json(
        "L6/stage6e_manifest.json",
        manifest,
    )

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
