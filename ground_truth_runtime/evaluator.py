from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


class GroundTruthValidationError(RuntimeError):
    pass


PROVIDERS = {
    "A": {"label": "TrOCR Sanskrit baseline", "manifest": "L5_provider_A/htr_manifest.json"},
    "B": {"label": "TrOCR Vedic Devanagari", "manifest": "L5_provider_B/htr_manifest.json"},
    "C": {"label": "Qwen visual HTR candidate", "manifest": "L5_provider_C/htr_manifest.json"},
}

ALLOWED_SOURCE_TEXT_STATUS = {
    "professor_supplied",
    "scholar_verified",
    "externally_verified",
    "verified_transcription",
}

ALLOWED_ALIGNMENT_STATUS = {
    "project_visual_reviewed",
    "scholar_verified",
    "externally_verified",
    "verified",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise GroundTruthValidationError(f"Required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GroundTruthValidationError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_outer(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", normalize_outer(value))


def content_stream(value: str) -> str:
    text = normalize_nfc(value)
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or category.startswith("Z") or category.startswith("P"):
            continue
        kept.append(char)
    return "".join(kept)


def levenshtein_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _stats(reference, hypothesis):
    distance = levenshtein_distance(reference, hypothesis)
    return {
        "distance": distance,
        "reference_units": len(reference),
        "rate": (
            distance / len(reference)
            if reference
            else (0.0 if not hypothesis else None)
        ),
    }


def strict_char_stats(reference: str, hypothesis: str) -> Dict[str, Any]:
    return _stats(list(normalize_nfc(reference)), list(normalize_nfc(hypothesis)))


def word_stats(reference: str, hypothesis: str) -> Dict[str, Any]:
    return _stats(normalize_nfc(reference).split(), normalize_nfc(hypothesis).split())


def content_char_stats(reference: str, hypothesis: str) -> Dict[str, Any]:
    return _stats(list(content_stream(reference)), list(content_stream(hypothesis)))


def evaluate_provider(
    references: Mapping[str, str],
    hypotheses: Mapping[str, str],
) -> Dict[str, Any]:
    ref_ids = sorted(references)
    hyp_ids = sorted(hypotheses)
    if ref_ids != hyp_ids:
        raise GroundTruthValidationError(
            "Provider/GT line IDs do not match exactly. "
            f"GT={ref_ids}, provider={hyp_ids}"
        )

    lines = []
    strict_edits = strict_ref = 0
    word_edits = word_ref = 0
    content_edits = content_ref = 0

    for line_id in ref_ids:
        reference = references[line_id]
        hypothesis = hypotheses[line_id]
        strict = strict_char_stats(reference, hypothesis)
        word = word_stats(reference, hypothesis)
        content = content_char_stats(reference, hypothesis)

        strict_edits += strict["distance"]
        strict_ref += strict["reference_units"]
        word_edits += word["distance"]
        word_ref += word["reference_units"]
        content_edits += content["distance"]
        content_ref += content["reference_units"]

        lines.append(
            {
                "line_id": line_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": strict["rate"],
                "wer": word["rate"],
                "content_cer": content["rate"],
                "char_edit_distance": strict["distance"],
                "char_reference_units": strict["reference_units"],
                "word_edit_distance": word["distance"],
                "word_reference_units": word["reference_units"],
                "content_char_edit_distance": content["distance"],
                "content_char_reference_units": content["reference_units"],
            }
        )

    return {
        "coverage": 1.0,
        "evaluated_line_count": len(lines),
        "strict_cer": strict_edits / strict_ref if strict_ref else None,
        "wer": word_edits / word_ref if word_ref else None,
        "content_cer": content_edits / content_ref if content_ref else None,
        "character_edit_distance": strict_edits,
        "character_reference_units": strict_ref,
        "word_edit_distance": word_edits,
        "word_reference_units": word_ref,
        "content_character_edit_distance": content_edits,
        "content_character_reference_units": content_ref,
        "lines": lines,
    }


def provider_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    provider = payload.get("provider")
    generation = payload.get("generation")

    provider_id = None
    model_id = None

    if isinstance(provider, Mapping):
        provider_id = (
            provider.get("provider_id")
            or provider.get("id")
            or provider.get("name")
        )
        model_id = provider.get("model_id") or provider.get("model")
    elif isinstance(provider, str):
        provider_id = provider

    if isinstance(generation, Mapping):
        model_id = model_id or generation.get("model_id") or generation.get("model")

    provider_id = provider_id or payload.get("provider_id") or payload.get("htr_provider")
    model_id = model_id or payload.get("model_id")

    return {"provider_id": provider_id, "model_id": model_id}


def load_provider_lines(manifest_path: Path) -> Dict[str, Any]:
    payload = _load_json(manifest_path)
    rows = payload.get("lines")
    if not isinstance(rows, list):
        raise GroundTruthValidationError(f"{manifest_path}: lines array missing.")

    line_texts = {}
    statuses = {}

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        line_id = row.get("line_id")
        if not isinstance(line_id, str) or not line_id.strip():
            continue

        text = row.get("devanagari_text")
        if not isinstance(text, str) or not text.strip():
            raw_script = row.get("raw_script")
            raw_text = row.get("raw_text")
            if raw_script == "devanagari" and isinstance(raw_text, str) and raw_text.strip():
                text = raw_text
            else:
                text = ""

        line_texts[line_id] = text
        statuses[line_id] = {
            "status": row.get("status"),
            "review_required": row.get("review_required"),
            "raw_script": row.get("raw_script"),
        }

    if not line_texts:
        raise GroundTruthValidationError(
            f"{manifest_path}: no line-level Devanagari hypotheses found."
        )

    return {
        "identity": provider_identity(payload),
        "line_texts": line_texts,
        "line_status": statuses,
    }


def _reference_lines(ground_truth: Mapping[str, Any]) -> Dict[str, str]:
    rows = ground_truth.get("lines")
    if not isinstance(rows, list):
        rows = ground_truth.get("physical_lines")
    if not isinstance(rows, list):
        raise GroundTruthValidationError("Ground Truth must contain a lines array.")

    references = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        line_id = row.get("line_id")
        text = row.get("reference_text")
        if not isinstance(line_id, str) or not line_id.strip():
            raise GroundTruthValidationError("Every Ground-Truth line requires line_id.")
        if not isinstance(text, str):
            raise GroundTruthValidationError(f"{line_id}: reference_text must be a string.")
        if line_id in references:
            raise GroundTruthValidationError(f"Duplicate Ground-Truth line_id: {line_id}")
        references[line_id] = text

    if not references:
        raise GroundTruthValidationError("Ground Truth contains no usable reference lines.")
    return references


def validate_ground_truth(
    ground_truth: Mapping[str, Any],
    provenance: Mapping[str, Any],
    expected_line_ids: Sequence[str],
) -> Dict[str, str]:
    source_status = str(ground_truth.get("source_text_status", "")).strip().lower()
    if source_status not in ALLOWED_SOURCE_TEXT_STATUS:
        raise GroundTruthValidationError(
            "source_text_status is not an allowed externally verified value."
        )

    alignment_status = str(
        ground_truth.get("physical_line_alignment_status", "")
    ).strip().lower()
    if alignment_status not in ALLOWED_ALIGNMENT_STATUS:
        raise GroundTruthValidationError(
            "physical_line_alignment_status is not an allowed reviewed value."
        )

    guardrails = ground_truth.get("scientific_guardrails")
    if not isinstance(guardrails, Mapping):
        raise GroundTruthValidationError("scientific_guardrails object is required.")
    if guardrails.get("cer_wer_allowed_for_stage5_transcription") is not True:
        raise GroundTruthValidationError(
            "CER/WER is blocked unless scientific_guardrails."
            "cer_wer_allowed_for_stage5_transcription is true."
        )

    if not isinstance(provenance, Mapping) or not provenance:
        raise GroundTruthValidationError("A non-empty provenance object is required.")

    golden_compatible = (
        isinstance(provenance.get("authoritative_source"), Mapping)
        and isinstance(provenance.get("derived_alignment"), Mapping)
    )

    verification = provenance.get("verification")
    generic_verified = False
    if isinstance(verification, Mapping):
        status = str(verification.get("status", "")).strip().lower()
        generic_verified = (
            status in {
                "verified",
                "scholar_verified",
                "professor_supplied",
                "externally_verified",
            }
            and bool(str(verification.get("verified_by_role", "")).strip())
        )

    if not golden_compatible and not generic_verified:
        raise GroundTruthValidationError(
            "Provenance must use authoritative_source + derived_alignment, "
            "or verification.status + verification.verified_by_role."
        )

    references = _reference_lines(ground_truth)
    expected = sorted(str(x) for x in expected_line_ids)
    actual = sorted(references)
    if actual != expected:
        raise GroundTruthValidationError(
            "Ground-Truth line IDs must exactly match Stage-5 provider line IDs. "
            f"Expected={expected}, GT={actual}"
        )
    return references


def build_stage5_benchmark(
    *,
    run_id: str,
    run_dir: Path,
    ground_truth: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    available = {}

    for key, config in PROVIDERS.items():
        path = run_dir / config["manifest"]
        if path.exists():
            available[key] = {
                "config": config,
                "loaded": load_provider_lines(path),
            }

    if not available:
        raise GroundTruthValidationError(
            "No Stage-5 Provider A/B/C manifests are available."
        )

    baseline_key = "A" if "A" in available else sorted(available)[0]
    expected_line_ids = sorted(
        available[baseline_key]["loaded"]["line_texts"]
    )
    references = validate_ground_truth(
        ground_truth,
        provenance,
        expected_line_ids,
    )

    providers = {}
    for key in ("A", "B", "C"):
        item = available.get(key)
        if not item:
            continue
        loaded = item["loaded"]
        evaluation = evaluate_provider(references, loaded["line_texts"])
        providers[key] = {
            "label": item["config"]["label"],
            "provider_id": loaded["identity"].get("provider_id"),
            "model_id": loaded["identity"].get("model_id"),
            "manifest": item["config"]["manifest"],
            **evaluation,
        }

    strict_ranking = sorted(
        providers,
        key=lambda key: (
            providers[key]["strict_cer"]
            if providers[key]["strict_cer"] is not None
            else float("inf")
        ),
    )
    content_ranking = sorted(
        providers,
        key=lambda key: (
            providers[key]["content_cer"]
            if providers[key]["content_cer"] is not None
            else float("inf")
        ),
    )

    dataset_id = ground_truth.get("dataset_id") or f"{run_id}_verified_gt_v1"

    return {
        "benchmark_version": "1.0-generic-run-ground-truth",
        "run_id": run_id,
        "benchmark_name": "Verified Ground-Truth Stage-5 Evaluation",
        "ground_truth_dataset": {
            "dataset_id": dataset_id,
            "canonical_path": "ground_truth/ground_truth.json",
            "provenance_path": "ground_truth/provenance.json",
            "source_text_status": ground_truth.get("source_text_status"),
            "physical_line_alignment": ground_truth.get(
                "physical_line_alignment_status"
            ),
            "translation_ground_truth_available": bool(
                ground_truth.get("translation_ground_truth_available", False)
            ),
        },
        "metric_policy": {
            "strict_cer": (
                "Unicode NFC code-point Levenshtein distance divided by "
                "verified-reference code points; whitespace and punctuation retained."
            ),
            "wer": (
                "Unicode NFC whitespace-token Levenshtein distance divided by "
                "verified-reference words; spacing/tokenization sensitive."
            ),
            "content_cer": (
                "Secondary diagnostic only; Unicode NFC then removal of "
                "whitespace/separators/punctuation; marks and spelling retained."
            ),
            "silent_sanskrit_correction": False,
            "transliteration_before_scoring": False,
            "H_is_accuracy": False,
            "T_is_accuracy": False,
            "H_or_T_replaced_by_CER_WER": False,
        },
        "providers": providers,
        "rankings": {
            "strict_cer_best_to_worst": strict_ranking,
            "content_cer_best_to_worst": content_ranking,
        },
        "scientific_interpretation": {
            "best_stage5_provider_by_strict_cer": (
                strict_ranking[0] if strict_ranking else None
            ),
            "best_stage5_provider_by_content_cer": (
                content_ranking[0] if content_ranking else None
            ),
            "H_remains_readiness_signal": True,
            "T_remains_stage6_trust_signal": True,
            "stage6_translation_validation_claim": False,
        },
    }


def attach_and_evaluate_ground_truth(
    *,
    run_id: str,
    run_dir: Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    ground_truth = payload.get("ground_truth") if isinstance(payload, Mapping) else None
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None

    if not isinstance(ground_truth, Mapping):
        raise GroundTruthValidationError("Request requires a ground_truth object.")
    if not isinstance(provenance, Mapping):
        raise GroundTruthValidationError("Request requires a provenance object.")

    benchmark = build_stage5_benchmark(
        run_id=run_id,
        run_dir=run_dir,
        ground_truth=ground_truth,
        provenance=provenance,
    )

    gt_dir = run_dir / "ground_truth"
    _write_json(gt_dir / "ground_truth.json", copy.deepcopy(dict(ground_truth)))
    _write_json(gt_dir / "provenance.json", copy.deepcopy(dict(provenance)))
    _write_json(gt_dir / "stage5_benchmark.json", benchmark)

    return {
        "run_id": run_id,
        "status": "verified_ground_truth_attached",
        "ground_truth_available": True,
        "dataset_id": benchmark["ground_truth_dataset"]["dataset_id"],
        "providers_evaluated": list(benchmark["providers"]),
        "best_provider_by_strict_cer": benchmark[
            "scientific_interpretation"
        ]["best_stage5_provider_by_strict_cer"],
        "artifacts": {
            "ground_truth": "ground_truth/ground_truth.json",
            "provenance": "ground_truth/provenance.json",
            "stage5_benchmark": "ground_truth/stage5_benchmark.json",
        },
        "scientific_guardrail": (
            "CER/WER/content-CER evaluate Stage-5 transcription only. "
            "Canonical H(A+B), A+C/B+C readiness evidence, T, routing thresholds "
            "and translation-validation status are unchanged."
        ),
        "benchmark": benchmark,
    }


def load_ground_truth_bundle(run_dir: Path) -> Dict[str, Any]:
    gt_dir = run_dir / "ground_truth"
    result = {
        "ground_truth": None,
        "provenance": None,
        "stage5_benchmark": None,
    }

    for key, name in (
        ("ground_truth", "ground_truth.json"),
        ("provenance", "provenance.json"),
        ("stage5_benchmark", "stage5_benchmark.json"),
    ):
        path = gt_dir / name
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                result[key] = None

    return result
