from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.artifact_store import ArtifactStore


LAYER6D_VERSION = "0.1.0-deterministic-context-retrieval"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6D input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6D corpus: {path}")

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"Stage 6D corpus is empty: {path}")

    return rows


def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    text = " ".join((text or "").split())

    if len(text) < n:
        return Counter({text: 1}) if text else Counter()

    return Counter(
        text[i : i + n]
        for i in range(
            0,
            len(text) - n + 1,
        )
    )


def _cosine(
    a: Counter[str],
    b: Counter[str],
) -> float:
    if not a or not b:
        return 0.0

    dot = sum(
        value * b.get(key, 0)
        for key, value in a.items()
    )

    if dot <= 0:
        return 0.0

    norm_a = math.sqrt(
        sum(value * value for value in a.values())
    )
    norm_b = math.sqrt(
        sum(value * value for value in b.values())
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        dot / (norm_a * norm_b)
    )


def _token_overlap(
    query: str,
    passage: str,
) -> float:
    q = {
        token
        for token in query.split()
        if token
    }

    p = {
        token
        for token in passage.split()
        if token
    }

    if not q:
        return 0.0

    return len(q & p) / float(len(q))


def _query_rows(
    stage6c_line: Dict[str, Any],
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []

    for row in stage6c_line.get(
        "sequence_morphology",
        [],
    ):
        segmented = str(
            row.get(
                "vidyut_segmented_readable",
                "",
            )
            or ""
        )

        if not segmented:
            continue

        queries.append(
            {
                "provider": row.get("provider"),
                "hypothesis_rank": row.get(
                    "hypothesis_rank"
                ),
                "query_devanagari": segmented,
                "morphology_coverage": row.get(
                    "morphology_coverage"
                ),
                "derivational_coverage": row.get(
                    "derivational_coverage"
                ),
            }
        )

    return queries


def _to_slp1(
    text: str,
    lipi: Any,
) -> str:
    if not text:
        return ""

    return lipi.transliterate(
        text,
        lipi.Scheme.Devanagari,
        lipi.Scheme.Slp1,
    )


def _score_passage(
    *,
    query_slp1: str,
    passage: Dict[str, Any],
    query_grams: Counter[str],
) -> Dict[str, float]:
    surface = str(
        passage.get(
            "surface_slp1",
            "",
        )
        or ""
    )

    lemma = str(
        passage.get(
            "lemma_slp1",
            "",
        )
        or ""
    )

    surface_ngram = _cosine(
        query_grams,
        _char_ngrams(surface, 3),
    )

    lemma_ngram = _cosine(
        query_grams,
        _char_ngrams(lemma, 3),
    )

    exact_token_overlap = _token_overlap(
        query_slp1,
        surface,
    )

    retrieval_score = (
        0.62 * surface_ngram
        + 0.23 * lemma_ngram
        + 0.15 * exact_token_overlap
    )

    return {
        "surface_char3_cosine": round(
            surface_ngram,
            4,
        ),
        "lemma_char3_cosine": round(
            lemma_ngram,
            4,
        ),
        "exact_token_overlap": round(
            exact_token_overlap,
            4,
        ),
        "retrieval_score": round(
            retrieval_score,
            4,
        ),
    }


def _retrieve(
    *,
    query: Dict[str, Any],
    corpus: Sequence[Dict[str, Any]],
    lipi: Any,
    top_k: int,
) -> Dict[str, Any]:
    query_deva = str(
        query.get(
            "query_devanagari",
            "",
        )
        or ""
    )

    query_slp1 = _to_slp1(
        query_deva,
        lipi,
    )

    query_grams = _char_ngrams(
        query_slp1,
        3,
    )

    scored: List[Dict[str, Any]] = []

    for passage in corpus:
        scores = _score_passage(
            query_slp1=query_slp1,
            passage=passage,
            query_grams=query_grams,
        )

        if scores["retrieval_score"] <= 0:
            continue

        scored.append(
            {
                "passage_id": passage.get(
                    "passage_id"
                ),
                "source_file": passage.get(
                    "source_file"
                ),
                "surface_slp1": passage.get(
                    "surface_slp1"
                ),
                "surface_devanagari": passage.get(
                    "surface_devanagari"
                ),
                "lemma_slp1": passage.get(
                    "lemma_slp1"
                ),
                "lemma_devanagari": passage.get(
                    "lemma_devanagari"
                ),
                "token_count": passage.get(
                    "token_count"
                ),
                **scores,
            }
        )

    scored.sort(
        key=lambda row: row[
            "retrieval_score"
        ],
        reverse=True,
    )

    return {
        **query,
        "query_slp1": query_slp1,
        "hits": scored[:top_k],
    }


def _aggregate_line_hits(
    retrievals: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_passage: Dict[
        Tuple[Any, Any],
        Dict[str, Any],
    ] = {}

    for retrieval in retrievals:
        for hit in retrieval.get(
            "hits",
            [],
        ):
            key = (
                hit.get("source_file"),
                hit.get("passage_id"),
            )

            existing = by_passage.get(key)

            evidence = {
                "provider": retrieval.get(
                    "provider"
                ),
                "hypothesis_rank": retrieval.get(
                    "hypothesis_rank"
                ),
                "query_devanagari": retrieval.get(
                    "query_devanagari"
                ),
                "retrieval_score": hit.get(
                    "retrieval_score"
                ),
            }

            if existing is None:
                by_passage[key] = {
                    **hit,
                    "query_support": [
                        evidence
                    ],
                }
            else:
                existing[
                    "query_support"
                ].append(
                    evidence
                )

                if (
                    float(
                        hit.get(
                            "retrieval_score",
                            0.0,
                        )
                    )
                    >
                    float(
                        existing.get(
                            "retrieval_score",
                            0.0,
                        )
                    )
                ):
                    for field in (
                        "retrieval_score",
                        "surface_char3_cosine",
                        "lemma_char3_cosine",
                        "exact_token_overlap",
                    ):
                        existing[field] = (
                            hit.get(field)
                        )

    rows = list(
        by_passage.values()
    )

    for row in rows:
        independent_providers = {
            support.get(
                "provider"
            )
            for support in row[
                "query_support"
            ]
            if support.get(
                "provider"
            )
        }

        row[
            "independent_provider_query_support"
        ] = len(
            independent_providers
        )

        row[
            "query_support_count"
        ] = len(
            row[
                "query_support"
            ]
        )

        best = float(
            row.get(
                "retrieval_score",
                0.0,
            )
        )

        # Retrieval corroboration is intentionally small; related N-best queries
        # are correlated evidence.
        corroboration_bonus = min(
            0.08,
            0.03
            * max(
                0,
                row[
                    "query_support_count"
                ]
                - 1,
            )
            + 0.02
            * max(
                0,
                row[
                    "independent_provider_query_support"
                ]
                - 1,
            ),
        )

        row[
            "context_evidence_score"
        ] = round(
            min(
                1.0,
                best
                + corroboration_bonus,
            ),
            4,
        )

        if (
            row[
                "context_evidence_score"
            ]
            >= 0.72
            and row[
                "independent_provider_query_support"
            ]
            >= 2
        ):
            row[
                "context_status"
            ] = (
                "strong_retrieval_corroboration"
            )
        elif (
            row[
                "context_evidence_score"
            ]
            >= 0.50
        ):
            row[
                "context_status"
            ] = (
                "moderate_retrieval_evidence"
            )
        else:
            row[
                "context_status"
            ] = (
                "weak_retrieval_evidence"
            )

        row[
            "promotion_to_final_text_allowed"
        ] = False

    rows.sort(
        key=lambda row: row[
            "context_evidence_score"
        ],
        reverse=True,
    )

    return rows


def run_layer6d_contextual_retrieval(
    store: ArtifactStore,
    *,
    corpus_path: str = (
        "knowledge/stage6d_dcs/passages.jsonl"
    ),
    top_k_per_query: int = 5,
    top_k_per_line: int = 10,
) -> Dict[str, Any]:
    """
    Stage 6D v0.1 — deterministic retrieval evidence.

    This is deliberately retrieval-only. It does not ask an LLM to rewrite HTR.
    The corpus is used to provide attested contextual evidence with provenance.
    """
    try:
        from vidyut import lipi
    except Exception as exc:
        raise RuntimeError(
            "Activate venv-stage6; Vidyut transliteration is required."
        ) from exc

    run_dir = Path(
        store.run_dir
    )

    stage6c = _load_json(
        run_dir
        / "L6"
        / "stage6c_manifest.json"
    )

    corpus = _load_jsonl(
        Path(corpus_path)
    )

    lines = stage6c.get(
        "lines"
    )

    if not isinstance(
        lines,
        list,
    ) or not lines:
        raise RuntimeError(
            "Stage 6D requires Stage 6C line results."
        )

    line_results: List[
        Dict[str, Any]
    ] = []

    total_queries = 0
    strong_retrieval_lines = 0
    moderate_retrieval_lines = 0

    for line in sorted(
        lines,
        key=lambda row: int(
            row.get(
                "reading_order",
                10**9,
            )
        ),
    ):
        queries = _query_rows(
            line
        )

        retrievals = [
            _retrieve(
                query=query,
                corpus=corpus,
                lipi=lipi,
                top_k=top_k_per_query,
            )
            for query in queries
        ]

        total_queries += len(
            retrievals
        )

        aggregated = (
            _aggregate_line_hits(
                retrievals
            )[:top_k_per_line]
        )

        best_context_score = (
            float(
                aggregated[0][
                    "context_evidence_score"
                ]
            )
            if aggregated
            else 0.0
        )

        best_context_status = (
            aggregated[0][
                "context_status"
            ]
            if aggregated
            else "no_retrieval_evidence"
        )

        if best_context_status == (
            "strong_retrieval_corroboration"
        ):
            strong_retrieval_lines += 1

        elif best_context_status == (
            "moderate_retrieval_evidence"
        ):
            moderate_retrieval_lines += 1

        line_results.append(
            {
                "line_id": line.get(
                    "line_id"
                ),
                "reading_order": line.get(
                    "reading_order"
                ),
                "H": line.get("H"),
                "cross_provider_agreement": (
                    line.get(
                        "cross_provider_agreement"
                    )
                ),
                "line_conclusion_from_stage6c": (
                    line.get(
                        "line_conclusion"
                    )
                ),
                "retrieval_queries": retrievals,
                "aggregated_context_hits": (
                    aggregated
                ),
                "best_context_evidence_score": round(
                    best_context_score,
                    4,
                ),
                "best_context_status": (
                    best_context_status
                ),
                "promotion_to_final_text_allowed": False,
            }
        )

    metrics = {
        "algorithm_version": (
            LAYER6D_VERSION
        ),
        "stage6_substage": (
            "6D_contextual_rag_disambiguation"
        ),
        "retrieval_mode": (
            "deterministic_char_ngram_plus_token_overlap"
        ),
        "corpus_passages": len(
            corpus
        ),
        "num_lines": len(
            line_results
        ),
        "retrieval_queries": (
            total_queries
        ),
        "strong_retrieval_lines": (
            strong_retrieval_lines
        ),
        "moderate_retrieval_lines": (
            moderate_retrieval_lines
        ),
        "generated_unobserved_sanskrit_tokens": 0,
        "llm_generation_used": False,
        "embedding_model_used": False,
        "final_diplomatic_transcription_available": False,
        "final_normalized_devanagari_available": False,
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
            "6D_contextual_rag_disambiguation"
        ),
        "version": (
            LAYER6D_VERSION
        ),
        "run_id": store.run_id,
        "status": (
            "context_retrieval_evidence_prepared"
        ),
        "corpus": {
            "path": corpus_path,
            "passages": len(
                corpus
            ),
            "expected_source": (
                "Ambuda sanitized DCS data"
            ),
        },
        "safety_contract": {
            "retrieved_text_is_evidence_not_ground_truth": True,
            "retrieval_may_not_override_low_H": True,
            "llm_free_rewrite_allowed": False,
            "unobserved_sanskrit_generation_allowed": False,
            "final_transcription_claim_allowed": False,
        },
        "metrics": metrics,
        "lines": line_results,
        "next_action": (
            "run_stage6e_evidence_constrained_reconstruction"
        ),
        "audit_note": (
            "Stage 6D retrieves attested Sanskrit passages with source/passsage "
            "provenance using deterministic noisy-string retrieval. Retrieval "
            "evidence is not final manuscript truth and cannot independently "
            "promote a reconstruction."
        ),
    }

    store.write_json(
        "L6/context_retrieval.json",
        {
            "version": (
                LAYER6D_VERSION
            ),
            "run_id": store.run_id,
            "lines": line_results,
        },
    )

    store.write_json(
        "L6/stage6d_manifest.json",
        manifest,
    )

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
