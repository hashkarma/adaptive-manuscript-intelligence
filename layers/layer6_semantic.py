from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.artifact_store import ArtifactStore


LAYER6_VERSION = "0.1.0-htr-evidence-parser"


@dataclass
class Stage6Hypothesis:
    provider: str
    rank: int
    exact_devanagari: str
    readable_devanagari: str
    raw_text: str
    relative_score: Optional[float]
    sequence_score: Optional[float]


@dataclass
class Stage6LineEvidence:
    line_id: str
    reading_order: int
    bbox: Dict[str, int]
    crop_rel_path: Optional[str]

    htr_readiness_H: Optional[float]
    cross_provider_agreement: Optional[float]
    reconstruction_mode: str
    evidence_status: str

    provider_a_top1_exact: str
    provider_a_top1_readable: str
    provider_b_top1_exact: str
    provider_b_top1_readable: str

    provider_a_hypotheses: List[Stage6Hypothesis]
    provider_b_hypotheses: List[Stage6Hypothesis]

    shared_readable_tokens: List[str]
    notes: List[str]


@dataclass
class Layer6Output:
    lines: List[Stage6LineEvidence]
    metrics: Dict[str, Any]
    manifest: Dict[str, Any]


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required Stage-6 input artifact is missing: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _readable_devanagari(text: str) -> str:
    """
    Readability normalization for Stage-6 NLP preparation.

    Exact HTR output remains preserved separately.

    This removes characters that are useful as raw palaeographic/Vedic evidence
    but often interfere with generic downstream tokenization:
      - Vedic Extensions U+1CD0..U+1CFF
      - Devanagari Extended accent/sign range U+A8E0..U+A8FF
      - residual generic combining diacritics U+0300..U+036F
      - replacement / square placeholder characters
      - invisible control / format characters

    This is NOT linguistic correction and does not invent characters.
    """
    text = unicodedata.normalize("NFC", text or "")
    kept: List[str] = []

    for ch in text:
        cp = ord(ch)

        if ch in {"\n", "\t"}:
            kept.append(ch)
            continue

        if (
            0x1CD0 <= cp <= 0x1CFF
            or 0xA8E0 <= cp <= 0xA8FF
            or 0x0300 <= cp <= 0x036F
            or 0xFE00 <= cp <= 0xFE0F
            or cp in {0x25A1, 0x25A0, 0xFFFD}
        ):
            continue

        if unicodedata.category(ch).startswith("C"):
            continue

        kept.append(ch)

    return " ".join("".join(kept).split())


def _tokens(text: str) -> List[str]:
    cleaned = (
        text.replace("।", " ")
        .replace("॥", " ")
        .replace("/", " ")
        .replace("|", " ")
    )

    return [
        token
        for token in cleaned.split()
        if token
    ]


def _shared_tokens(a: str, b: str) -> List[str]:
    """
    Conservative lexical overlap only.

    We deliberately do not fuzzy-correct tokens here. Stage 6A is an evidence
    parser, not a Sanskrit correction engine.
    """
    a_tokens = _tokens(a)
    b_tokens = set(_tokens(b))

    shared: List[str] = []

    for token in a_tokens:
        if len(token) < 2:
            continue

        if token in b_tokens and token not in shared:
            shared.append(token)

    return shared


def _extract_hypotheses(
    line: Dict[str, Any],
    *,
    provider_name: str,
) -> List[Stage6Hypothesis]:
    """
    Accept the Stage-5 schemas used across the prototype.

    Top-1 is always preserved even if a manifest stores alternatives under a
    different key.
    """
    hypotheses: List[Stage6Hypothesis] = []

    top1_exact = str(
        line.get(
            "devanagari_text",
            line.get("devanagari", ""),
        )
        or ""
    )

    top1_raw = str(
        line.get(
            "raw_text",
            line.get("raw_iast", ""),
        )
        or ""
    )

    hypotheses.append(
        Stage6Hypothesis(
            provider=provider_name,
            rank=1,
            exact_devanagari=top1_exact,
            readable_devanagari=_readable_devanagari(top1_exact),
            raw_text=top1_raw,
            relative_score=None,
            sequence_score=None,
        )
    )

    source = line.get("hypotheses")

    if not isinstance(source, list):
        source = line.get("alternatives")

    if not isinstance(source, list):
        source = []

    for index, item in enumerate(source, start=1):
        if not isinstance(item, dict):
            continue

        rank = int(item.get("rank", index))

        if rank == 1:
            # Prefer the explicit top-level transcription as canonical top-1.
            continue

        exact = str(
            item.get(
                "devanagari_text",
                item.get(
                    "devanagari",
                    item.get("text", ""),
                ),
            )
            or ""
        )

        raw = str(
            item.get(
                "raw_text",
                item.get(
                    "raw_iast",
                    item.get("text", ""),
                ),
            )
            or ""
        )

        hypotheses.append(
            Stage6Hypothesis(
                provider=provider_name,
                rank=rank,
                exact_devanagari=exact,
                readable_devanagari=_readable_devanagari(exact),
                raw_text=raw,
                relative_score=_float_or_none(
                    item.get("relative_score")
                ),
                sequence_score=_float_or_none(
                    item.get("sequence_score")
                ),
            )
        )

    hypotheses.sort(key=lambda item: item.rank)

    return hypotheses


def _manifest_lines(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for row in manifest.get("lines", []):
        if not isinstance(row, dict):
            continue

        line_id = str(row.get("line_id", "") or "")

        if line_id:
            result[line_id] = row

    return result


def _comparison_lines(comparison: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for row in comparison.get("lines", []):
        if not isinstance(row, dict):
            continue

        line_id = str(row.get("line_id", "") or "")

        if line_id:
            result[line_id] = row

    return result


def _readiness_lines(readiness: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Support both:
      {"lines": [...]}
    and
      {"line_readiness": [...]}
    style artifacts.
    """
    source = readiness.get("lines")

    if not isinstance(source, list):
        source = readiness.get("line_readiness")

    if not isinstance(source, list):
        source = []

    result: Dict[str, Dict[str, Any]] = {}

    for row in source:
        if not isinstance(row, dict):
            continue

        line_id = str(row.get("line_id", "") or "")

        if line_id:
            result[line_id] = row

    return result


def _line_H(
    row: Dict[str, Any],
    page_H: Optional[float],
) -> Optional[float]:
    for key in (
        "htr_readiness_H",
        "H",
        "H_line",
        "htr_readiness",
    ):
        value = _float_or_none(row.get(key))

        if value is not None:
            return value

    return page_H


def _line_agreement(
    comparison_row: Dict[str, Any],
) -> Optional[float]:
    for key in (
        "content_char_similarity",
        "cross_provider_agreement",
        "strict_char_similarity",
    ):
        value = _float_or_none(comparison_row.get(key))

        if value is not None:
            return value

    return None


def _reconstruction_mode(H: Optional[float]) -> str:
    if H is None:
        return "deep_semantic_reconstruction"

    if H >= 0.75:
        return "standard_semantic_processing"

    if H >= 0.50:
        return "enhanced_semantic_reconstruction"

    return "deep_semantic_reconstruction"


def _evidence_status(
    H: Optional[float],
    agreement: Optional[float],
) -> str:
    if H is None:
        return "uncalibrated_htr_evidence"

    if H >= 0.75 and (agreement is None or agreement >= 0.75):
        return "higher_trust_htr_evidence"

    if H >= 0.50:
        return "intermediate_htr_evidence"

    return "low_trust_htr_evidence"


def run_layer6_htr_evidence_parser(
    store: ArtifactStore,
    *,
    provider_a_dir: str = "L5_provider_A",
    provider_b_dir: str = "L5_provider_B",
    comparison_dir: str = "L5_compare",
    readiness_dir: str = "L5_readiness",
) -> Layer6Output:
    """
    Stage 6A — HTR evidence ingestion and logical sequence reconstruction.

    This implements the first Stage-6 responsibility from the semantic proposal:
    ingest noisy HTR output, preserve reading order, and create a normalized
    representation for later Viccheda/morphology/RAG processing.

    It deliberately DOES NOT:
      - invent corrected Sanskrit,
      - run Viccheda,
      - translate,
      - calculate final semantic trust T(p),
      - route to a scholar.

    Human review remains deferred until the later Stage-6 reconstruction and
    semantic-validation steps have been attempted.
    """
    run_dir = Path(store.run_dir)

    stage4 = _load_json(
        run_dir / "L4" / "line_manifest.json"
    )

    manifest_a = _load_json(
        run_dir / provider_a_dir / "htr_manifest.json"
    )

    manifest_b = _load_json(
        run_dir / provider_b_dir / "htr_manifest.json"
    )

    comparison = _load_json(
        run_dir / comparison_dir / "provider_comparison.json"
    )

    readiness = _load_json(
        run_dir / readiness_dir / "htr_readiness.json"
    )

    a_lines = _manifest_lines(manifest_a)
    b_lines = _manifest_lines(manifest_b)
    c_lines = _comparison_lines(comparison)
    h_lines = _readiness_lines(readiness)

    page = readiness.get("page", {}) or {}
    page_H = _float_or_none(
        page.get(
            "htr_readiness_H_page",
            page.get("H_page"),
        )
    )

    lines: List[Stage6LineEvidence] = []

    for stage4_line in sorted(
        stage4.get("lines", []),
        key=lambda row: int(row.get("reading_order", 10**9)),
    ):
        line_id = str(stage4_line.get("line_id", "") or "")

        if not line_id:
            continue

        a = a_lines.get(line_id, {})
        b = b_lines.get(line_id, {})
        c = c_lines.get(line_id, {})
        hrow = h_lines.get(line_id, {})

        a_hypotheses = _extract_hypotheses(
            a,
            provider_name="provider_a",
        )

        b_hypotheses = _extract_hypotheses(
            b,
            provider_name="provider_b",
        )

        a_exact = (
            a_hypotheses[0].exact_devanagari
            if a_hypotheses
            else ""
        )

        a_readable = (
            a_hypotheses[0].readable_devanagari
            if a_hypotheses
            else ""
        )

        b_exact = (
            b_hypotheses[0].exact_devanagari
            if b_hypotheses
            else ""
        )

        b_readable = (
            b_hypotheses[0].readable_devanagari
            if b_hypotheses
            else ""
        )

        H = _line_H(hrow, page_H)
        agreement = _line_agreement(c)

        notes: List[str] = []

        if H is not None and H < 0.50:
            notes.append("LOW_HTR_READINESS_DO_NOT_TRUST_TOP1")

        if agreement is not None and agreement < 0.50:
            notes.append("LOW_CROSS_PROVIDER_AGREEMENT")

        if not a_exact:
            notes.append("PROVIDER_A_TEXT_MISSING")

        if not b_exact:
            notes.append("PROVIDER_B_TEXT_MISSING")

        shared = _shared_tokens(
            a_readable,
            b_readable,
        )

        lines.append(
            Stage6LineEvidence(
                line_id=line_id,
                reading_order=int(
                    stage4_line.get(
                        "reading_order",
                        len(lines) + 1,
                    )
                ),
                bbox={
                    "x": int(stage4_line.get("x", 0)),
                    "y": int(stage4_line.get("y", 0)),
                    "w": int(stage4_line.get("w", 0)),
                    "h": int(stage4_line.get("h", 0)),
                },
                crop_rel_path=stage4_line.get("crop_rel_path"),
                htr_readiness_H=H,
                cross_provider_agreement=agreement,
                reconstruction_mode=_reconstruction_mode(H),
                evidence_status=_evidence_status(H, agreement),
                provider_a_top1_exact=a_exact,
                provider_a_top1_readable=a_readable,
                provider_b_top1_exact=b_exact,
                provider_b_top1_readable=b_readable,
                provider_a_hypotheses=a_hypotheses,
                provider_b_hypotheses=b_hypotheses,
                shared_readable_tokens=shared,
                notes=notes,
            )
        )

    if not lines:
        raise RuntimeError(
            "Stage 6A could not construct any ordered HTR evidence lines."
        )

    low_htr_lines = sum(
        1
        for line in lines
        if (
            line.htr_readiness_H is None
            or line.htr_readiness_H < 0.50
        )
    )

    low_agreement_lines = sum(
        1
        for line in lines
        if (
            line.cross_provider_agreement is not None
            and line.cross_provider_agreement < 0.50
        )
    )

    deep_lines = sum(
        1
        for line in lines
        if line.reconstruction_mode == "deep_semantic_reconstruction"
    )

    metrics: Dict[str, Any] = {
        "algorithm_version": LAYER6_VERSION,
        "stage6_substage": "6A_htr_evidence_parser",
        "num_lines": len(lines),
        "page_htr_readiness_H": page_H,
        "low_htr_readiness_lines": low_htr_lines,
        "low_cross_provider_agreement_lines": low_agreement_lines,
        "deep_reconstruction_lines": deep_lines,
        "standard_processing_lines": sum(
            1
            for line in lines
            if line.reconstruction_mode == "standard_semantic_processing"
        ),
        "enhanced_reconstruction_lines": sum(
            1
            for line in lines
            if line.reconstruction_mode == "enhanced_semantic_reconstruction"
        ),
        "final_semantic_trust_T": None,
        "translation_available": False,
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_exhausted": True,
    }

    manifest: Dict[str, Any] = {
        "stage": "stage6_semantic_interpretation",
        "substage": "6A_htr_evidence_parser",
        "version": LAYER6_VERSION,
        "run_id": store.run_id,
        "status": "evidence_prepared",
        "input_contract": {
            "stage4_line_manifest": "L4/line_manifest.json",
            "provider_a_manifest": f"{provider_a_dir}/htr_manifest.json",
            "provider_b_manifest": f"{provider_b_dir}/htr_manifest.json",
            "provider_comparison": f"{comparison_dir}/provider_comparison.json",
            "htr_readiness": f"{readiness_dir}/htr_readiness.json",
        },
        "output_contract": {
            "next_substage": "6B_viccheda_and_candidate_reconstruction",
            "final_readable_devanagari_available": False,
            "translation_available": False,
            "semantic_trust_T_available": False,
        },
        "human_review_policy": {
            "stage5_low_H_is_not_a_human_stop": True,
            "scholar_review_deferred": True,
            "scholar_review_condition": (
                "Only after Stage 6 reconstruction, semantic validation and "
                "configured adaptive retries remain unresolved."
            ),
        },
        "lines": [asdict(line) for line in lines],
        "metrics": metrics,
        "next_action": "run_stage6_viccheda_and_candidate_reconstruction",
        "audit_note": (
            "Stage 6A preserves exact HTR output and creates a readability-only "
            "normalization for NLP preparation. It performs no Sanskrit correction "
            "and makes no translation or scholarly-correctness claim."
        ),
    }

    store.write_json(
        "L6/htr_input_table.json",
        {
            "version": LAYER6_VERSION,
            "run_id": store.run_id,
            "rows": [asdict(line) for line in lines],
        },
    )

    csv_path = Path(
        store.path("L6/htr_input_table.csv")
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "line_id",
                "reading_order",
                "x",
                "y",
                "w",
                "h",
                "crop_rel_path",
                "H",
                "cross_provider_agreement",
                "reconstruction_mode",
                "evidence_status",
                "provider_a_top1_readable",
                "provider_b_top1_readable",
                "shared_readable_tokens",
            ],
        )

        writer.writeheader()

        for line in lines:
            writer.writerow(
                {
                    "line_id": line.line_id,
                    "reading_order": line.reading_order,
                    "x": line.bbox["x"],
                    "y": line.bbox["y"],
                    "w": line.bbox["w"],
                    "h": line.bbox["h"],
                    "crop_rel_path": line.crop_rel_path,
                    "H": line.htr_readiness_H,
                    "cross_provider_agreement": line.cross_provider_agreement,
                    "reconstruction_mode": line.reconstruction_mode,
                    "evidence_status": line.evidence_status,
                    "provider_a_top1_readable": line.provider_a_top1_readable,
                    "provider_b_top1_readable": line.provider_b_top1_readable,
                    "shared_readable_tokens": " | ".join(
                        line.shared_readable_tokens
                    ),
                }
            )

    store.write_json(
        "L6/page_sequence_candidates.json",
        {
            "version": LAYER6_VERSION,
            "run_id": store.run_id,
            "provider_a_page_readable": "\n".join(
                line.provider_a_top1_readable
                for line in lines
            ),
            "provider_b_page_readable": "\n".join(
                line.provider_b_top1_readable
                for line in lines
            ),
            "line_order": [
                line.line_id
                for line in lines
            ],
            "note": (
                "These are alternative HTR evidence streams. Neither is the "
                "final Stage-6 reconstructed manuscript text."
            ),
        },
    )

    store.write_json(
        "L6/stage6_manifest.json",
        manifest,
    )

    return Layer6Output(
        lines=lines,
        metrics=metrics,
        manifest=manifest,
    )
