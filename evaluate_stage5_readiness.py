from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional


READINESS_VERSION = "0.1.0-stage5f-htr-readiness"

# Weighted geometric fusion.
#
# The design deliberately gives the largest weights to:
#   - decoder reliability
#   - cross-provider agreement
#
# so that clean Unicode/script output cannot hide weak recognition evidence.
WEIGHTS = {
    "segmentation_readiness": 0.10,
    "completion": 0.05,
    "script_integrity": 0.05,
    "decoder_reliability": 0.30,
    "sequence_quality": 0.15,
    "cross_provider_agreement": 0.35,
}

EPSILON = 0.01


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _safe_geometric_mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [
        _clip01(float(value))
        for value in values
        if value is not None
    ]
    if not clean:
        return None

    return math.exp(
        sum(math.log(max(value, EPSILON)) for value in clean)
        / len(clean)
    )


def _weighted_geometric_mean(
    evidence: Dict[str, float],
) -> float:
    """
    Weighted geometric fusion prevents strong signals from fully compensating
    for a critically weak signal.

    This is intentional: low decoder reliability or low cross-provider
    agreement should not be hidden by clean script output or high segmentation
    readiness.
    """
    total = 0.0

    for name, weight in WEIGHTS.items():
        value = _clip01(evidence[name])
        total += weight * math.log(max(value, EPSILON))

    return _clip01(math.exp(total))


def _read_stage4_segmentation_readiness(
    run_dir: str,
) -> float:
    path = os.path.join(
        run_dir,
        "L4",
        "line_manifest.json",
    )

    manifest = _load_json(path)

    metrics = manifest.get("metrics", {})
    value = _safe_float(
        metrics.get("segmentation_confidence")
    )

    if value is None:
        raise ValueError(
            "Stage 4 segmentation_confidence is unavailable."
        )

    return _clip01(value)


def _extract_lines(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rows = manifest.get("lines", [])
    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        line_id = str(row.get("line_id", "")).strip()

        if line_id:
            result[line_id] = row

    return result


def _quality(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    value = row.get("quality", {})
    return value if isinstance(value, dict) else {}


def _completion_evidence(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> float:
    ok_a = str(row_a.get("status", "")).lower() == "ok"
    ok_b = str(row_b.get("status", "")).lower() == "ok"

    if ok_a and ok_b:
        return 1.0

    if ok_a or ok_b:
        return 0.60

    return 0.0


def _provider_script_integrity(
    row: Dict[str, Any],
) -> Optional[float]:
    quality = _quality(row)

    purity = _safe_float(
        quality.get("script_purity")
    )

    if purity is None:
        return None

    purity = _clip01(purity)

    transliteration_valid = quality.get(
        "transliteration_valid"
    )

    # Older Provider-A artifacts may have explicit False for one malformed
    # transliteration while still having high script purity.
    #
    # Do not collapse such a line to zero: malformed post-processing is weaker
    # evidence than total recognition failure, but it must still be penalized.
    if transliteration_valid is False:
        purity = min(purity, 0.85)

    return purity


def _script_integrity_evidence(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> float:
    value = _safe_geometric_mean(
        [
            _provider_script_integrity(row_a),
            _provider_script_integrity(row_b),
        ]
    )

    return 0.50 if value is None else value


def _provider_decoder_reliability(
    row: Dict[str, Any],
) -> Optional[float]:
    quality = _quality(row)

    stability = _safe_float(
        quality.get("decoder_stability")
    )

    if stability is not None:
        return _clip01(stability)

    # Fallback only when stability is unavailable:
    # low normalized entropy implies stronger differentiation among N-best.
    entropy = _safe_float(
        quality.get("hypothesis_entropy")
    )

    if entropy is not None:
        return _clip01(1.0 - entropy)

    return None


def _decoder_reliability_evidence(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> float:
    value = _safe_mean(
        [
            _provider_decoder_reliability(row_a),
            _provider_decoder_reliability(row_b),
        ]
    )

    # Missing decoder evidence is not treated as confidence.
    return 0.25 if value is None else _clip01(value)


def _provider_sequence_quality(
    row: Dict[str, Any],
) -> Optional[float]:
    quality = _quality(row)

    repetition_ratio = _safe_float(
        quality.get("token_repetition_ratio")
    )

    length_spread = _safe_float(
        quality.get("hypothesis_length_spread")
    )

    anomaly_score = _safe_float(
        quality.get("sequence_length_anomaly_score")
    )

    factors: List[Optional[float]] = []

    if repetition_ratio is not None:
        factors.append(
            1.0 - _clip01(repetition_ratio)
        )

    if length_spread is not None:
        # Smoothly penalize large N-best length disagreement.
        factors.append(
            1.0 / (1.0 + max(0.0, length_spread))
        )

    if anomaly_score is not None:
        factors.append(
            1.0 - _clip01(anomaly_score)
        )

    return _safe_geometric_mean(factors)


def _sequence_quality_evidence(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> float:
    value = _safe_geometric_mean(
        [
            _provider_sequence_quality(row_a),
            _provider_sequence_quality(row_b),
        ]
    )

    return 0.50 if value is None else value


def _cross_provider_agreement(
    comparison_row: Dict[str, Any],
) -> float:
    value = _safe_float(
        comparison_row.get(
            "content_char_similarity"
        )
    )

    return 0.0 if value is None else _clip01(value)


def _route_line(
    *,
    H: float,
    segmentation_readiness: float,
    completion: float,
    script_integrity: float,
    decoder_reliability: float,
    sequence_quality: float,
    cross_provider_agreement: float,
) -> Dict[str, Any]:
    """
    Provisional Stage-5F routing rules.

    These are engineering/orchestration rules, not scholarly truth rules.
    """

    reasons: List[str] = []

    if segmentation_readiness < 0.60:
        reasons.append("LOW_STAGE4_SEGMENTATION_READINESS")
        return {
            "decision": "return_to_stage4",
            "reason_codes": reasons,
        }

    if completion < 1.0:
        reasons.append("INCOMPLETE_HTR_PROVIDER_COVERAGE")
        return {
            "decision": "retry_or_switch_htr_provider",
            "reason_codes": reasons,
        }

    if script_integrity < 0.90:
        reasons.append("LOW_SCRIPT_OR_TRANSLITERATION_INTEGRITY")
        return {
            "decision": "repair_or_review_htr_output",
            "reason_codes": reasons,
        }

    if (
        cross_provider_agreement < 0.50
        and decoder_reliability < 0.25
    ):
        reasons.extend(
            [
                "LOW_CROSS_PROVIDER_AGREEMENT",
                "LOW_DECODER_RELIABILITY",
            ]
        )

        return {
            "decision": "third_provider_or_scholar_review",
            "reason_codes": reasons,
        }

    if sequence_quality < 0.45:
        reasons.append("LOW_SEQUENCE_QUALITY")

    if (
        H >= 0.75
        and cross_provider_agreement >= 0.75
        and decoder_reliability >= 0.55
        and sequence_quality >= 0.60
    ):
        return {
            "decision": "provisionally_proceed_to_stage6",
            "reason_codes": [],
        }

    if H >= 0.50:
        reasons.append("INTERMEDIATE_HTR_READINESS")

        return {
            "decision": "contextual_rerank_or_additional_provider",
            "reason_codes": reasons,
        }

    reasons.append("LOW_HTR_READINESS")

    return {
        "decision": "third_provider_or_scholar_review",
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def evaluate_readiness(
    *,
    stage4_S: float,
    manifest_a: Dict[str, Any],
    manifest_b: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:

    lines_a = _extract_lines(manifest_a)
    lines_b = _extract_lines(manifest_b)

    comparison_rows = {
        str(row.get("line_id")): row
        for row in comparison.get("lines", [])
        if isinstance(row, dict)
        and row.get("line_id") is not None
    }

    line_ids = sorted(
        set(lines_a)
        | set(lines_b)
        | set(comparison_rows),
        key=lambda line_id: int(
            lines_a.get(
                line_id,
                lines_b.get(
                    line_id,
                    {},
                ),
            ).get(
                "reading_order",
                10**9,
            )
        ),
    )

    line_results: List[Dict[str, Any]] = []

    for line_id in line_ids:
        row_a = lines_a.get(line_id, {})
        row_b = lines_b.get(line_id, {})
        compare_row = comparison_rows.get(
            line_id,
            {},
        )

        completion = _completion_evidence(
            row_a,
            row_b,
        )

        script_integrity = (
            _script_integrity_evidence(
                row_a,
                row_b,
            )
        )

        decoder_reliability = (
            _decoder_reliability_evidence(
                row_a,
                row_b,
            )
        )

        sequence_quality = (
            _sequence_quality_evidence(
                row_a,
                row_b,
            )
        )

        cross_agreement = (
            _cross_provider_agreement(
                compare_row
            )
        )

        evidence = {
            "segmentation_readiness": (
                stage4_S
            ),
            "completion": (
                completion
            ),
            "script_integrity": (
                script_integrity
            ),
            "decoder_reliability": (
                decoder_reliability
            ),
            "sequence_quality": (
                sequence_quality
            ),
            "cross_provider_agreement": (
                cross_agreement
            ),
        }

        H_line = _weighted_geometric_mean(
            evidence
        )

        route = _route_line(
            H=H_line,
            segmentation_readiness=stage4_S,
            completion=completion,
            script_integrity=script_integrity,
            decoder_reliability=decoder_reliability,
            sequence_quality=sequence_quality,
            cross_provider_agreement=cross_agreement,
        )

        reading_order = int(
            row_a.get(
                "reading_order",
                row_b.get(
                    "reading_order",
                    0,
                ),
            )
        )

        line_results.append(
            {
                "line_id": line_id,
                "reading_order": reading_order,
                "evidence": {
                    key: round(value, 4)
                    for key, value in evidence.items()
                },
                "htr_readiness_H_line": round(
                    H_line,
                    4,
                ),
                "routing": route,
                "interpretation": (
                    "System HTR readiness only. "
                    "Not recognition accuracy."
                ),
            }
        )

    H_lines = [
        float(row["htr_readiness_H_line"])
        for row in line_results
    ]

    agreement_values = [
        float(
            row["evidence"][
                "cross_provider_agreement"
            ]
        )
        for row in line_results
    ]

    decoder_values = [
        float(
            row["evidence"][
                "decoder_reliability"
            ]
        )
        for row in line_results
    ]

    sequence_values = [
        float(
            row["evidence"][
                "sequence_quality"
            ]
        )
        for row in line_results
    ]

    completion_values = [
        float(
            row["evidence"][
                "completion"
            ]
        )
        for row in line_results
    ]

    script_values = [
        float(
            row["evidence"][
                "script_integrity"
            ]
        )
        for row in line_results
    ]

    page_evidence = {
        "segmentation_readiness": stage4_S,
        "completion": (
            _safe_mean(completion_values)
            or 0.0
        ),
        "script_integrity": (
            _safe_mean(script_values)
            or 0.0
        ),
        "decoder_reliability": (
            _safe_mean(decoder_values)
            or 0.0
        ),
        "sequence_quality": (
            _safe_mean(sequence_values)
            or 0.0
        ),
        "cross_provider_agreement": (
            _safe_mean(agreement_values)
            or 0.0
        ),
    }

    H_page = _weighted_geometric_mean(
        page_evidence
    )

    low_line_count = sum(
        1
        for value in H_lines
        if value < 0.50
    )

    low_agreement_count = sum(
        1
        for value in agreement_values
        if value < 0.50
    )

    page_route = _route_line(
        H=H_page,
        segmentation_readiness=page_evidence[
            "segmentation_readiness"
        ],
        completion=page_evidence[
            "completion"
        ],
        script_integrity=page_evidence[
            "script_integrity"
        ],
        decoder_reliability=page_evidence[
            "decoder_reliability"
        ],
        sequence_quality=page_evidence[
            "sequence_quality"
        ],
        cross_provider_agreement=page_evidence[
            "cross_provider_agreement"
        ],
    )

    # Page-level safety rule:
    # never auto-proceed when more than 20% of lines have low agreement.
    if (
        len(line_results) > 0
        and low_agreement_count
        / len(line_results)
        > 0.20
    ):
        page_route = {
            "decision": (
                "third_provider_or_scholar_review"
            ),
            "reason_codes": list(
                dict.fromkeys(
                    page_route.get(
                        "reason_codes",
                        [],
                    )
                    + [
                        "PAGE_WIDE_LOW_CROSS_PROVIDER_AGREEMENT"
                    ]
                )
            ),
        }

    return {
        "readiness_version": READINESS_VERSION,

        "definition": {
            "H": (
                "HTR system readiness for downstream processing; "
                "NOT recognition accuracy."
            ),
            "fusion_method": (
                "weighted_geometric_mean"
            ),
            "weights": WEIGHTS,
            "reason": (
                "Geometric fusion prevents high segmentation/script "
                "scores from hiding low decoder reliability or low "
                "cross-provider agreement."
            ),
            "ground_truth_dependency": (
                "CER/WER remain unavailable until scholar-verified "
                "ground truth exists."
            ),
        },

        "page": {
            "evidence": {
                key: round(value, 4)
                for key, value in page_evidence.items()
            },

            "htr_readiness_H_page": round(
                H_page,
                4,
            ),

            "line_H_mean": (
                round(
                    statistics.mean(H_lines),
                    4,
                )
                if H_lines
                else None
            ),

            "line_H_median": (
                round(
                    statistics.median(H_lines),
                    4,
                )
                if H_lines
                else None
            ),

            "low_readiness_lines": (
                low_line_count
            ),

            "low_agreement_lines": (
                low_agreement_count
            ),

            "total_lines": (
                len(line_results)
            ),

            "routing": (
                page_route
            ),

            "cer": None,
            "wer": None,
            "ground_truth_available": False,
            "recognition_accuracy_calibrated": False,
        },

        "lines": (
            line_results
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5F: calculate provisional HTR readiness H(p) "
            "from frozen Provider-A/B evidence and cross-provider "
            "agreement."
        )
    )

    parser.add_argument(
        "run_id"
    )

    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )

    parser.add_argument(
        "--provider-a-dir",
        default="L5_provider_A",
    )

    parser.add_argument(
        "--provider-b-dir",
        default="L5_provider_B_11line_final",
    )

    parser.add_argument(
        "--comparison-dir",
        default="L5_compare",
    )

    args = parser.parse_args()

    run_dir = os.path.join(
        args.artifacts,
        args.run_id,
    )

    stage4_path = os.path.join(
        run_dir,
        "L4",
        "line_manifest.json",
    )

    provider_a_path = os.path.join(
        run_dir,
        args.provider_a_dir,
        "htr_manifest.json",
    )

    provider_b_path = os.path.join(
        run_dir,
        args.provider_b_dir,
        "htr_manifest.json",
    )

    comparison_path = os.path.join(
        run_dir,
        args.comparison_dir,
        "provider_comparison.json",
    )

    for path in [
        stage4_path,
        provider_a_path,
        provider_b_path,
        comparison_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact is missing: {path}"
            )

    print("Stage 5F version:", READINESS_VERSION)
    print("Provider A:", provider_a_path)
    print("Provider B:", provider_b_path)
    print("Comparison:", comparison_path)

    stage4_S = (
        _read_stage4_segmentation_readiness(
            run_dir
        )
    )

    manifest_a = _load_json(
        provider_a_path
    )

    manifest_b = _load_json(
        provider_b_path
    )

    comparison = _load_json(
        comparison_path
    )

    output = evaluate_readiness(
        stage4_S=stage4_S,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        comparison=comparison,
    )

    output_dir = os.path.join(
        run_dir,
        "L5_readiness",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "htr_readiness.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPage readiness:")

    page = output["page"]

    for name, value in page[
        "evidence"
    ].items():
        print(
            f"  {name}: {_fmt(value)}"
        )

    print(
        "  H_page:",
        _fmt(
            page[
                "htr_readiness_H_page"
            ]
        ),
    )

    print(
        "  low_readiness_lines:",
        page["low_readiness_lines"],
    )

    print(
        "  low_agreement_lines:",
        page["low_agreement_lines"],
    )

    print(
        "  routing:",
        page["routing"]["decision"],
    )

    print(
        "  reason_codes:",
        ", ".join(
            page["routing"][
                "reason_codes"
            ]
        )
        or "none",
    )

    print("\nPer-line HTR readiness:")

    for row in output["lines"]:
        print()
        print(
            f"[{row['line_id']}] "
            f"H={row['htr_readiness_H_line']:.4f} "
            f"route={row['routing']['decision']}"
        )

        evidence = row[
            "evidence"
        ]

        print(
            "  S={:.4f} completion={:.4f} "
            "script={:.4f} decoder={:.4f} "
            "sequence={:.4f} agreement={:.4f}".format(
                evidence[
                    "segmentation_readiness"
                ],
                evidence[
                    "completion"
                ],
                evidence[
                    "script_integrity"
                ],
                evidence[
                    "decoder_reliability"
                ],
                evidence[
                    "sequence_quality"
                ],
                evidence[
                    "cross_provider_agreement"
                ],
            )
        )

        print(
            "  reasons:",
            ", ".join(
                row["routing"][
                    "reason_codes"
                ]
            )
            or "none",
        )

    print(
        "\nReadiness artifact:",
        output_path,
    )

    print(
        "\nIMPORTANT: H(p) is system readiness evidence only. "
        "It is not OCR/HTR accuracy. CER/WER remain unset."
    )


if __name__ == "__main__":
    main()
