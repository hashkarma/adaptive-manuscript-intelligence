from __future__ import annotations

import argparse
import json
import os
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


COMPARE_VERSION = "0.1.0-stage5e-cross-provider-agreement"


@dataclass
class LineAgreement:
    line_id: str
    reading_order: int

    provider_a_status: str
    provider_b_status: str

    provider_a_devanagari: str
    provider_b_devanagari: str

    strict_char_distance: Optional[int]
    strict_char_similarity: Optional[float]

    content_char_distance: Optional[int]
    content_char_similarity: Optional[float]

    provider_a_length: int
    provider_b_length: int
    length_ratio: Optional[float]

    provider_a_entropy: Optional[float]
    provider_b_entropy: Optional[float]

    provider_a_decoder_stability: Optional[float]
    provider_b_decoder_stability: Optional[float]

    provider_a_script_purity: Optional[float]
    provider_b_script_purity: Optional[float]

    provider_a_token_repetition_ratio: Optional[float]
    provider_b_token_repetition_ratio: Optional[float]

    provider_a_hypothesis_length_spread: Optional[float]
    provider_b_hypothesis_length_spread: Optional[float]

    notes: List[str]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_path(run_dir: str, snapshot_dir: str) -> str:
    path = os.path.join(
        run_dir,
        snapshot_dir,
        "htr_manifest.json",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find HTR manifest: {path}"
        )
    return path


def _normalise_strict(text: str) -> str:
    """
    Conservative normalization:
      - NFC
      - collapse whitespace
      - preserve punctuation
      - preserve Vedic/Devanagari marks
    """
    return " ".join(
        unicodedata.normalize("NFC", text or "")
        .strip()
        .split()
    )


def _normalise_content(text: str) -> str:
    """
    Content-oriented comparison:
      - NFC
      - remove whitespace
      - remove Unicode punctuation
      - retain letters, digits and combining marks

    This is agreement evidence only, never an accuracy measure.
    """
    text = unicodedata.normalize("NFC", text or "")

    kept: List[str] = []

    for ch in text:
        category = unicodedata.category(ch)

        if ch.isspace():
            continue

        if category.startswith("P"):
            continue

        kept.append(ch)

    return "".join(kept)


def _levenshtein_distance(a: str, b: str) -> int:
    """
    Memory-efficient Levenshtein distance.
    """
    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    previous = list(range(len(a) + 1))

    for row_index, b_char in enumerate(b, start=1):
        current = [row_index]

        for col_index, a_char in enumerate(a, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            substitute_cost = (
                previous[col_index - 1]
                + (0 if a_char == b_char else 1)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost,
                )
            )

        previous = current

    return previous[-1]


def _similarity(a: str, b: str) -> Tuple[int, float]:
    distance = _levenshtein_distance(a, b)

    denominator = max(
        len(a),
        len(b),
    )

    if denominator == 0:
        return 0, 1.0

    similarity = 1.0 - (
        distance / denominator
    )

    return (
        distance,
        round(
            max(
                0.0,
                min(
                    1.0,
                    similarity,
                ),
            ),
            4,
        ),
    )


def _safe_float(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(
    values: List[Optional[float]],
) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return round(
        sum(clean) / len(clean),
        4,
    )


def _safe_median(
    values: List[Optional[float]],
) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return round(
        statistics.median(clean),
        4,
    )


def _extract_lines(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    lines = manifest.get(
        "lines",
        [],
    )

    if not isinstance(lines, list):
        raise ValueError(
            "HTR manifest 'lines' must be a list."
        )

    result: Dict[str, Dict[str, Any]] = {}

    for row in lines:
        if not isinstance(row, dict):
            continue

        line_id = str(
            row.get("line_id", "")
        ).strip()

        if line_id:
            result[line_id] = row

    return result


def _extract_devanagari(
    row: Dict[str, Any],
) -> str:
    """
    Supports both the older Provider-A artifact schema and the newer
    Provider-B schema.
    """
    candidates = [
        row.get("devanagari_text"),
        row.get("transcription_devanagari"),
        row.get("devanagari"),
    ]

    for value in candidates:
        if isinstance(value, str):
            return value

    return ""


def _extract_quality(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    quality = row.get(
        "quality",
        {},
    )

    return (
        quality
        if isinstance(quality, dict)
        else {}
    )


def _length_ratio(
    length_a: int,
    length_b: int,
) -> Optional[float]:
    maximum = max(
        length_a,
        length_b,
    )

    if maximum == 0:
        return 1.0

    minimum = min(
        length_a,
        length_b,
    )

    return round(
        minimum / maximum,
        4,
    )


def compare_manifests(
    manifest_a: Dict[str, Any],
    manifest_b: Dict[str, Any],
) -> Dict[str, Any]:

    lines_a = _extract_lines(
        manifest_a
    )

    lines_b = _extract_lines(
        manifest_b
    )

    all_line_ids = sorted(
        set(lines_a)
        | set(lines_b),
        key=lambda line_id: (
            int(
                (
                    lines_a.get(
                        line_id,
                        lines_b.get(
                            line_id,
                            {},
                        ),
                    )
                    .get(
                        "reading_order",
                        10**9,
                    )
                )
            ),
            line_id,
        ),
    )

    comparisons: List[
        LineAgreement
    ] = []

    for line_id in all_line_ids:

        row_a = lines_a.get(
            line_id,
            {},
        )

        row_b = lines_b.get(
            line_id,
            {},
        )

        status_a = str(
            row_a.get(
                "status",
                "missing",
            )
        )

        status_b = str(
            row_b.get(
                "status",
                "missing",
            )
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

        dev_a_raw = _extract_devanagari(
            row_a
        )

        dev_b_raw = _extract_devanagari(
            row_b
        )

        dev_a = _normalise_strict(
            dev_a_raw
        )

        dev_b = _normalise_strict(
            dev_b_raw
        )

        content_a = _normalise_content(
            dev_a_raw
        )

        content_b = _normalise_content(
            dev_b_raw
        )

        notes: List[str] = []

        if status_a != "ok":
            notes.append(
                "PROVIDER_A_NOT_OK"
            )

        if status_b != "ok":
            notes.append(
                "PROVIDER_B_NOT_OK"
            )

        if (
            status_a == "ok"
            and status_b == "ok"
        ):
            strict_distance, strict_similarity = (
                _similarity(
                    dev_a,
                    dev_b,
                )
            )

            content_distance, content_similarity = (
                _similarity(
                    content_a,
                    content_b,
                )
            )

        else:
            strict_distance = None
            strict_similarity = None
            content_distance = None
            content_similarity = None

        if (
            content_similarity is not None
            and content_similarity < 0.50
        ):
            notes.append(
                "LOW_CROSS_PROVIDER_AGREEMENT"
            )

        elif (
            content_similarity is not None
            and content_similarity < 0.75
        ):
            notes.append(
                "MODERATE_CROSS_PROVIDER_AGREEMENT"
            )

        quality_a = _extract_quality(
            row_a
        )

        quality_b = _extract_quality(
            row_b
        )

        comparisons.append(
            LineAgreement(
                line_id=line_id,
                reading_order=reading_order,

                provider_a_status=status_a,
                provider_b_status=status_b,

                provider_a_devanagari=dev_a,
                provider_b_devanagari=dev_b,

                strict_char_distance=strict_distance,
                strict_char_similarity=strict_similarity,

                content_char_distance=content_distance,
                content_char_similarity=content_similarity,

                provider_a_length=len(
                    content_a
                ),
                provider_b_length=len(
                    content_b
                ),
                length_ratio=_length_ratio(
                    len(content_a),
                    len(content_b),
                ),

                provider_a_entropy=_safe_float(
                    quality_a.get(
                        "hypothesis_entropy"
                    )
                ),
                provider_b_entropy=_safe_float(
                    quality_b.get(
                        "hypothesis_entropy"
                    )
                ),

                provider_a_decoder_stability=_safe_float(
                    quality_a.get(
                        "decoder_stability"
                    )
                ),
                provider_b_decoder_stability=_safe_float(
                    quality_b.get(
                        "decoder_stability"
                    )
                ),

                provider_a_script_purity=_safe_float(
                    quality_a.get(
                        "script_purity"
                    )
                ),
                provider_b_script_purity=_safe_float(
                    quality_b.get(
                        "script_purity"
                    )
                ),

                provider_a_token_repetition_ratio=_safe_float(
                    quality_a.get(
                        "token_repetition_ratio"
                    )
                ),
                provider_b_token_repetition_ratio=_safe_float(
                    quality_b.get(
                        "token_repetition_ratio"
                    )
                ),

                provider_a_hypothesis_length_spread=_safe_float(
                    quality_a.get(
                        "hypothesis_length_spread"
                    )
                ),
                provider_b_hypothesis_length_spread=_safe_float(
                    quality_b.get(
                        "hypothesis_length_spread"
                    )
                ),

                notes=notes,
            )
        )

    strict_scores = [
        row.strict_char_similarity
        for row in comparisons
    ]

    content_scores = [
        row.content_char_similarity
        for row in comparisons
    ]

    length_ratios = [
        row.length_ratio
        for row in comparisons
    ]

    comparable = [
        row
        for row in comparisons
        if row.content_char_similarity
        is not None
    ]

    low_agreement = [
        row
        for row in comparable
        if (
            row.content_char_similarity
            is not None
            and row.content_char_similarity
            < 0.50
        )
    ]

    moderate_agreement = [
        row
        for row in comparable
        if (
            row.content_char_similarity
            is not None
            and 0.50
            <= row.content_char_similarity
            < 0.75
        )
    ]

    high_agreement = [
        row
        for row in comparable
        if (
            row.content_char_similarity
            is not None
            and row.content_char_similarity
            >= 0.75
        )
    ]

    provider_a_meta = manifest_a.get(
        "provider",
        {},
    )

    provider_b_meta = manifest_b.get(
        "provider",
        {},
    )

    output = {
        "comparison_version": COMPARE_VERSION,

        "provider_a": {
            "provider_id": (
                provider_a_meta.get(
                    "provider_id"
                )
                or manifest_a.get(
                    "provider"
                )
                or manifest_a.get(
                    "metrics",
                    {},
                ).get(
                    "provider"
                )
            ),
            "model_id": (
                provider_a_meta.get(
                    "model_id"
                )
                or manifest_a.get(
                    "model_id"
                )
                or manifest_a.get(
                    "metrics",
                    {},
                ).get(
                    "model_id"
                )
            ),
            "stage5_version": (
                manifest_a.get(
                    "version"
                )
            ),
            "metrics": (
                manifest_a.get(
                    "metrics",
                    {},
                )
            ),
        },

        "provider_b": {
            "provider_id": (
                provider_b_meta.get(
                    "provider_id"
                )
                or manifest_b.get(
                    "provider"
                )
                or manifest_b.get(
                    "metrics",
                    {},
                ).get(
                    "provider"
                )
            ),
            "model_id": (
                provider_b_meta.get(
                    "model_id"
                )
                or manifest_b.get(
                    "model_id"
                )
                or manifest_b.get(
                    "metrics",
                    {},
                ).get(
                    "model_id"
                )
            ),
            "stage5_version": (
                manifest_b.get(
                    "version"
                )
            ),
            "metrics": (
                manifest_b.get(
                    "metrics",
                    {},
                )
            ),
        },

        "agreement_definition": {
            "strict_char_similarity": (
                "1 - Levenshtein distance / max length "
                "after NFC + whitespace collapse; punctuation preserved."
            ),
            "content_char_similarity": (
                "1 - Levenshtein distance / max length "
                "after NFC with whitespace and Unicode punctuation removed; "
                "Devanagari/Vedic letters and marks preserved."
            ),
            "interpretation": (
                "Cross-provider agreement is evidence of consistency only. "
                "It is NOT recognition accuracy and MUST NOT substitute "
                "for scholar ground truth, CER, or WER."
            ),
            "provisional_bands": {
                "low": "< 0.50",
                "moderate": "0.50 to < 0.75",
                "high": ">= 0.75",
                "note": (
                    "Bands are diagnostic only and are not yet H(p) thresholds."
                ),
            },
        },

        "aggregate": {
            "total_lines": len(
                comparisons
            ),
            "comparable_lines": len(
                comparable
            ),

            "mean_strict_char_similarity": (
                _safe_mean(
                    strict_scores
                )
            ),
            "median_strict_char_similarity": (
                _safe_median(
                    strict_scores
                )
            ),

            "mean_content_char_similarity": (
                _safe_mean(
                    content_scores
                )
            ),
            "median_content_char_similarity": (
                _safe_median(
                    content_scores
                )
            ),

            "mean_length_ratio": (
                _safe_mean(
                    length_ratios
                )
            ),
            "median_length_ratio": (
                _safe_median(
                    length_ratios
                )
            ),

            "low_agreement_lines": len(
                low_agreement
            ),
            "moderate_agreement_lines": len(
                moderate_agreement
            ),
            "high_agreement_lines": len(
                high_agreement
            ),

            "htr_readiness_H": None,
            "cer": None,
            "wer": None,
            "ground_truth_available": False,
        },

        "lines": [
            asdict(
                row
            )
            for row in comparisons
        ],
    }

    return output


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return "N/A"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.4f}"

    return str(value)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Stage 5E: compare two frozen HTR provider snapshots "
            "without calculating H(p)."
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
        help=(
            "Snapshot directory inside the run directory "
            "for Provider A."
        ),
    )

    parser.add_argument(
        "--provider-b-dir",
        default="L5_provider_B_11line_final",
        help=(
            "Snapshot directory inside the run directory "
            "for Provider B."
        ),
    )

    args = parser.parse_args()

    run_dir = os.path.join(
        args.artifacts,
        args.run_id,
    )

    manifest_a_path = _manifest_path(
        run_dir,
        args.provider_a_dir,
    )

    manifest_b_path = _manifest_path(
        run_dir,
        args.provider_b_dir,
    )

    print("Stage 5E version:", COMPARE_VERSION)
    print("Provider A manifest:", manifest_a_path)
    print("Provider B manifest:", manifest_b_path)

    manifest_a = _load_json(
        manifest_a_path
    )

    manifest_b = _load_json(
        manifest_b_path
    )

    comparison = compare_manifests(
        manifest_a,
        manifest_b,
    )

    output_dir = os.path.join(
        run_dir,
        "L5_compare",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "provider_comparison.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            comparison,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nAggregate comparison:")

    aggregate = comparison[
        "aggregate"
    ]

    for key, value in aggregate.items():
        print(
            f"  {key}: {_fmt(value)}"
        )

    print("\nPer-line agreement:")

    for row in comparison["lines"]:
        print()
        print(
            f"[{row['line_id']}] "
            f"content={_fmt(row['content_char_similarity'])} "
            f"strict={_fmt(row['strict_char_similarity'])} "
            f"length_ratio={_fmt(row['length_ratio'])}"
        )

        if row["notes"]:
            print(
                "  notes:",
                ", ".join(
                    row["notes"]
                ),
            )

        print(
            "  A:",
            row["provider_a_devanagari"],
        )

        print(
            "  B:",
            row["provider_b_devanagari"],
        )

    print(
        "\nComparison artifact:",
        output_path,
    )

    print(
        "\nIMPORTANT: agreement is consistency evidence only; "
        "H(p), CER and WER remain unset."
    )


if __name__ == "__main__":
    main()
