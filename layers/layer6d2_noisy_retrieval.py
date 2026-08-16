from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rapidfuzz import fuzz, process

from core.artifact_store import ArtifactStore


LAYER6D2_VERSION = "0.1.0-noisy-surface-fuzzy-retrieval"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6D.2 input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6D corpus: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))

    if not rows:
        raise RuntimeError(f"Stage 6D corpus is empty: {path}")

    return rows


def _token_overlap(query: str, passage: str) -> float:
    q = {t for t in query.split() if t}
    p = {t for t in passage.split() if t}

    if not q:
        return 0.0

    return len(q & p) / float(len(q))


def _query_records(stage6d_line: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for row in stage6d_line.get("retrieval_queries", []):
        q_slp1 = str(row.get("query_slp1", "") or "")
        if not q_slp1:
            continue

        records.append(
            {
                "provider": row.get("provider"),
                "hypothesis_rank": row.get("hypothesis_rank"),
                "query_devanagari": row.get("query_devanagari"),
                "query_slp1": q_slp1,
                "morphology_coverage": row.get("morphology_coverage"),
                "derivational_coverage": row.get("derivational_coverage"),
            }
        )

    return records


def _retrieve_one(
    query: Dict[str, Any],
    *,
    corpus: Sequence[Dict[str, Any]],
    choices: Sequence[str],
    preselect_k: int,
    top_k: int,
) -> Dict[str, Any]:
    q = query["query_slp1"]

    matches = process.extract(
        q,
        choices,
        scorer=fuzz.partial_ratio,
        processor=None,
        limit=preselect_k,
        score_cutoff=20.0,
    )

    rescored: List[Dict[str, Any]] = []

    try:
        morphology_coverage = float(query.get("morphology_coverage") or 0.0)
    except (TypeError, ValueError):
        morphology_coverage = 0.0

    for _, partial_score, index in matches:
        passage = corpus[index]
        surface = choices[index]

        full_ratio = fuzz.ratio(q, surface) / 100.0
        wratio = fuzz.WRatio(q, surface) / 100.0
        partial = float(partial_score) / 100.0
        overlap = _token_overlap(q, surface)

        # Partial alignment dominates because HTR queries are short/noisy and
        # target phrases may occur inside a longer attested passage.
        #
        # Morphology is only a small query-quality prior. These weights are
        # engineering heuristics, not calibrated probabilities.
        noisy_surface_score = (
            0.58 * partial
            + 0.17 * wratio
            + 0.12 * full_ratio
            + 0.08 * overlap
            + 0.05 * morphology_coverage
        )

        rescored.append(
            {
                "source_file": passage.get("source_file"),
                "passage_id": passage.get("passage_id"),
                "surface_slp1": passage.get("surface_slp1"),
                "surface_devanagari": passage.get("surface_devanagari"),
                "lemma_slp1": passage.get("lemma_slp1"),
                "lemma_devanagari": passage.get("lemma_devanagari"),
                "partial_ratio": round(partial, 4),
                "wratio": round(wratio, 4),
                "full_ratio": round(full_ratio, 4),
                "exact_token_overlap": round(overlap, 4),
                "query_morphology_coverage": round(morphology_coverage, 4),
                "noisy_surface_score": round(noisy_surface_score, 4),
                "promotion_to_final_text_allowed": False,
            }
        )

    rescored.sort(
        key=lambda row: row["noisy_surface_score"],
        reverse=True,
    )

    return {
        **query,
        "hits": rescored[:top_k],
    }


def _aggregate_line_hits(
    query_results: Sequence[Dict[str, Any]],
    *,
    top_k_line: int,
) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}

    for result in query_results:
        provider = result.get("provider")
        rank = result.get("hypothesis_rank")

        for hit in result.get("hits", []):
            key = f"{hit.get('source_file')}::{hit.get('passage_id')}"

            evidence = {
                "provider": provider,
                "hypothesis_rank": rank,
                "query_devanagari": result.get("query_devanagari"),
                "query_slp1": result.get("query_slp1"),
                "noisy_surface_score": hit.get("noisy_surface_score"),
            }

            existing = by_key.get(key)

            if existing is None:
                existing = {
                    **hit,
                    "query_support": [],
                }
                by_key[key] = existing

            existing["query_support"].append(evidence)

            if float(hit.get("noisy_surface_score") or 0.0) > float(
                existing.get("noisy_surface_score") or 0.0
            ):
                for field in (
                    "partial_ratio",
                    "wratio",
                    "full_ratio",
                    "exact_token_overlap",
                    "query_morphology_coverage",
                    "noisy_surface_score",
                ):
                    existing[field] = hit.get(field)

    rows = list(by_key.values())

    for row in rows:
        providers = {
            e.get("provider")
            for e in row["query_support"]
            if e.get("provider")
        }

        supports = len(row["query_support"])
        provider_count = len(providers)

        # Corroboration is deliberately capped because N-best hypotheses from
        # one provider are correlated observations.
        bonus = min(
            0.08,
            0.015 * max(0, supports - 1)
            + 0.025 * max(0, provider_count - 1),
        )

        score = min(
            1.0,
            float(row.get("noisy_surface_score") or 0.0) + bonus,
        )

        row["query_support_count"] = supports
        row["independent_provider_query_support"] = provider_count
        row["fused_noisy_surface_score"] = round(score, 4)

        if score >= 0.78 and provider_count >= 2:
            status = "strong_noisy_surface_corroboration"
        elif score >= 0.62:
            status = "moderate_noisy_surface_evidence"
        else:
            status = "weak_noisy_surface_evidence"

        row["noisy_surface_status"] = status
        row["promotion_to_final_text_allowed"] = False

    rows.sort(
        key=lambda row: row["fused_noisy_surface_score"],
        reverse=True,
    )

    return rows[:top_k_line]


def run_layer6d2_noisy_surface_retrieval(
    store: ArtifactStore,
    *,
    corpus_path: str = "knowledge/stage6d_dcs/passages.jsonl",
    preselect_k: int = 60,
    top_k_query: int = 8,
    top_k_line: int = 12,
) -> Dict[str, Any]:
    """
    Stage 6D.2 — noisy-surface phrase retrieval.

    This stage exists because sentence embeddings are not expected to repair
    severe HTR corruption. It searches attested Sanskrit passages using local
    fuzzy alignment, then records provenance. It never rewrites the manuscript.
    """
    run_dir = Path(store.run_dir)

    stage6d = _load_json(
        run_dir / "L6" / "stage6d_manifest.json"
    )

    lines = stage6d.get("lines")
    if not isinstance(lines, list) or not lines:
        raise RuntimeError("Stage 6D.2 requires Stage 6D line results.")

    corpus = _load_jsonl(Path(corpus_path))
    choices = [
        str(row.get("surface_slp1", "") or "")
        for row in corpus
    ]

    line_results: List[Dict[str, Any]] = []

    total_queries = 0
    moderate_lines = 0
    strong_lines = 0

    for line in sorted(
        lines,
        key=lambda row: int(row.get("reading_order", 10**9)),
    ):
        queries = _query_records(line)

        query_results = [
            _retrieve_one(
                query,
                corpus=corpus,
                choices=choices,
                preselect_k=preselect_k,
                top_k=top_k_query,
            )
            for query in queries
        ]

        total_queries += len(query_results)

        aggregated = _aggregate_line_hits(
            query_results,
            top_k_line=top_k_line,
        )

        if aggregated:
            best_score = aggregated[0]["fused_noisy_surface_score"]
            best_status = aggregated[0]["noisy_surface_status"]
        else:
            best_score = 0.0
            best_status = "no_noisy_surface_evidence"

        if best_status == "strong_noisy_surface_corroboration":
            strong_lines += 1
        elif best_status == "moderate_noisy_surface_evidence":
            moderate_lines += 1

        line_results.append(
            {
                "line_id": line.get("line_id"),
                "reading_order": line.get("reading_order"),
                "H": line.get("H"),
                "cross_provider_agreement": line.get(
                    "cross_provider_agreement"
                ),
                "stage6d1_best_context_score": line.get(
                    "best_context_evidence_score"
                ),
                "stage6d1_best_context_status": line.get(
                    "best_context_status"
                ),
                "query_results": query_results,
                "aggregated_noisy_surface_hits": aggregated,
                "best_noisy_surface_score": round(float(best_score), 4),
                "best_noisy_surface_status": best_status,
                "promotion_to_final_text_allowed": False,
            }
        )

    metrics = {
        "algorithm_version": LAYER6D2_VERSION,
        "stage6_substage": "6D2_noisy_surface_fuzzy_retrieval",
        "corpus_passages": len(corpus),
        "num_lines": len(line_results),
        "retrieval_queries": total_queries,
        "strong_noisy_surface_lines": strong_lines,
        "moderate_noisy_surface_lines": moderate_lines,
        "rapidfuzz_partial_alignment_used": True,
        "embedding_model_used": False,
        "llm_generation_used": False,
        "generated_unobserved_sanskrit_tokens": 0,
        "final_diplomatic_transcription_available": False,
        "final_normalized_devanagari_available": False,
        "final_semantic_trust_T": None,
        "translation_available": False,
        "scholar_review_required": False,
        "scholar_review_deferred_until_stage6_exhausted": True,
    }

    manifest = {
        "stage": "stage6_semantic_interpretation",
        "substage": "6D2_noisy_surface_fuzzy_retrieval",
        "version": LAYER6D2_VERSION,
        "run_id": store.run_id,
        "status": "noisy_surface_retrieval_evidence_prepared",
        "corpus": {
            "path": corpus_path,
            "passages": len(corpus),
        },
        "safety_contract": {
            "retrieved_text_is_evidence_not_ground_truth": True,
            "retrieval_may_not_override_low_H": True,
            "semantic_embedding_not_used_for_raw_htr_repair": True,
            "unobserved_sanskrit_generation_allowed": False,
            "final_transcription_claim_allowed": False,
        },
        "metrics": metrics,
        "lines": line_results,
        "next_action": "evaluate_stage6d2_before_stage6e_candidate_generation",
        "audit_note": (
            "Stage 6D.2 uses RapidFuzz local partial alignment over attested "
            "SLP1 corpus passages to bridge noisy HTR and clean Sanskrit. "
            "It preserves source passage provenance and never promotes a "
            "retrieved string directly to manuscript transcription."
        ),
    }

    store.write_json(
        "L6/noisy_surface_retrieval.json",
        {
            "version": LAYER6D2_VERSION,
            "run_id": store.run_id,
            "lines": line_results,
        },
    )

    store.write_json(
        "L6/stage6d2_manifest.json",
        manifest,
    )

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
