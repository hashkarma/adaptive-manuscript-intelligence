from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.artifact_store import ArtifactStore


LAYER6C_VERSION = "0.1.0-conservative-morphology-grammar-validation"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6C input: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return default

    if callable(value):
        try:
            return value()
        except Exception:
            return default

    return value


def _enum_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    for attr in ("value", "name"):
        try:
            candidate = getattr(value, attr)
        except Exception:
            continue

        if callable(candidate):
            try:
                candidate = candidate()
            except Exception:
                continue

        if candidate is not None:
            return str(candidate)

    try:
        return str(value)
    except Exception:
        return None


def _serializable_repr(value: Any, limit: int = 800) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"

    if len(text) > limit:
        text = text[: limit - 3] + "..."

    return text


def _load_vidyut(
    data_root: str | Path,
) -> Tuple[Any, Any, Any]:
    try:
        from vidyut import lipi
        from vidyut.kosha import Kosha
        from vidyut.prakriya import Vyakarana
    except Exception as exc:
        raise RuntimeError(
            "Stage 6C requires Vidyut. Activate venv-stage6 first."
        ) from exc

    root = Path(data_root)
    kosha_path = root / "kosha"

    if not kosha_path.exists():
        raise FileNotFoundError(
            f"Missing Vidyut Kosha data: {kosha_path}"
        )

    kosha = Kosha(str(kosha_path))

    # nlp_mode preserves final s/r, which is useful when validating observed
    # NLP-style word forms. No Chandasi rules are enabled in this first pass.
    vyakarana = Vyakarana(
        log_steps=False,
        is_chandasi=False,
        use_svaras=False,
        nlp_mode=True,
    )

    return lipi, kosha, vyakarana


def _to_slp1(text: str, lipi: Any) -> str:
    if not text:
        return ""

    return lipi.transliterate(
        text,
        lipi.Scheme.Devanagari,
        lipi.Scheme.Slp1,
    )


def _to_devanagari(text: str, lipi: Any) -> str:
    if not text:
        return ""

    return lipi.transliterate(
        text,
        lipi.Scheme.Slp1,
        lipi.Scheme.Devanagari,
    )


def _kosha_entries(
    kosha: Any,
    slp1: str,
) -> List[Any]:
    if not slp1:
        return []

    try:
        entries = kosha.get(slp1)
    except Exception:
        return []

    try:
        return list(entries)
    except Exception:
        return []


def _entry_kind(entry: Any) -> str:
    """
    Avoid depending on the exact PyO3-generated class names.

    Subanta entries expose nominal features; Tinanta entries expose verbal
    features. We inspect those properties conservatively.
    """
    has_lakara = _safe_getattr(entry, "lakara") is not None
    has_purusha = _safe_getattr(entry, "purusha") is not None
    has_vibhakti = _safe_getattr(entry, "vibhakti") is not None

    if has_lakara or has_purusha:
        return "tinanta"

    if has_vibhakti:
        return "subanta"

    if bool(_safe_getattr(entry, "is_avyaya", False)):
        return "avyaya"

    class_name = type(entry).__name__.lower()

    if "tinanta" in class_name:
        return "tinanta"

    if "subanta" in class_name:
        return "subanta"

    return "unknown"


def _extract_entry_analysis(
    entry: Any,
    *,
    lipi: Any,
) -> Dict[str, Any]:
    kind = _entry_kind(entry)

    lemma_slp1 = _safe_getattr(entry, "lemma")
    if lemma_slp1 is not None:
        lemma_slp1 = str(lemma_slp1)

    analysis: Dict[str, Any] = {
        "entry_kind": kind,
        "lemma_slp1": lemma_slp1,
        "lemma_devanagari": (
            _to_devanagari(
                lemma_slp1,
                lipi,
            )
            if lemma_slp1
            else None
        ),
        "is_avyaya": bool(
            _safe_getattr(
                entry,
                "is_avyaya",
                False,
            )
        ),
        "linga": _enum_text(
            _safe_getattr(
                entry,
                "linga",
            )
        ),
        "vibhakti": _enum_text(
            _safe_getattr(
                entry,
                "vibhakti",
            )
        ),
        "vacana": _enum_text(
            _safe_getattr(
                entry,
                "vacana",
            )
        ),
        "prayoga": _enum_text(
            _safe_getattr(
                entry,
                "prayoga",
            )
        ),
        "lakara": _enum_text(
            _safe_getattr(
                entry,
                "lakara",
            )
        ),
        "purusha": _enum_text(
            _safe_getattr(
                entry,
                "purusha",
            )
        ),
        "entry_repr": _serializable_repr(
            entry
        ),
    }

    return analysis


def _derive_entry(
    entry: Any,
    *,
    observed_slp1: str,
    vyakarana: Any,
) -> Dict[str, Any]:
    try:
        derivations = vyakarana.derive(entry)
    except Exception as exc:
        return {
            "derive_succeeded": False,
            "derive_error": type(exc).__name__,
            "derived_forms_slp1": [],
            "exact_surface_rederived": False,
        }

    forms: List[str] = []

    for prakriya in derivations[:20]:
        text = _safe_getattr(
            prakriya,
            "text",
        )

        if isinstance(text, str) and text:
            forms.append(text)

    # Keep stable order without duplicates.
    unique_forms: List[str] = []
    seen = set()

    for form in forms:
        if form in seen:
            continue

        seen.add(form)
        unique_forms.append(form)

    return {
        "derive_succeeded": True,
        "derive_error": None,
        "derived_forms_slp1": unique_forms,
        "exact_surface_rederived": (
            observed_slp1
            in unique_forms
        ),
    }


def _analyse_surface(
    slp1: str,
    *,
    kosha: Any,
    vyakarana: Any,
    lipi: Any,
) -> Dict[str, Any]:
    entries = _kosha_entries(
        kosha,
        slp1,
    )

    analyses: List[Dict[str, Any]] = []
    exact_rederived = False

    for index, entry in enumerate(
        entries[:25],
        start=1,
    ):
        entry_analysis = (
            _extract_entry_analysis(
                entry,
                lipi=lipi,
            )
        )

        derivation = _derive_entry(
            entry,
            observed_slp1=slp1,
            vyakarana=vyakarana,
        )

        exact_rederived = (
            exact_rederived
            or derivation[
                "exact_surface_rederived"
            ]
        )

        analyses.append(
            {
                "analysis_index": index,
                **entry_analysis,
                **derivation,
            }
        )

    entry_kinds = sorted(
        {
            analysis["entry_kind"]
            for analysis in analyses
        }
    )

    lemmas_slp1 = sorted(
        {
            analysis["lemma_slp1"]
            for analysis in analyses
            if analysis.get("lemma_slp1")
        }
    )

    return {
        "slp1": slp1,
        "devanagari": _to_devanagari(
            slp1,
            lipi,
        ),
        "kosha_entry_count": len(entries),
        "morphologically_analyzable": bool(entries),
        "exact_surface_rederived": bool(
            exact_rederived
        ),
        "entry_kinds": entry_kinds,
        "lemmas_slp1": lemmas_slp1,
        "lemmas_devanagari": [
            _to_devanagari(
                lemma,
                lipi,
            )
            for lemma in lemmas_slp1
        ],
        "analyses": analyses,
    }


def _cluster_validation_status(
    *,
    cluster: Dict[str, Any],
    morphology: Dict[str, Any],
    H: Optional[float],
    agreement: Optional[float],
) -> str:
    """
    Morphology may establish linguistic plausibility, but it cannot overcome
    weak visual evidence.

    Therefore Stage 6C does not promote a candidate to manuscript truth.
    """
    morph = bool(
        morphology.get(
            "morphologically_analyzable"
        )
    )

    rederived = bool(
        morphology.get(
            "exact_surface_rederived"
        )
    )

    cross_provider = bool(
        cluster.get(
            "cross_provider_support"
        )
    )

    try:
        h = float(H)
    except (TypeError, ValueError):
        h = 0.0

    try:
        a = float(agreement)
    except (TypeError, ValueError):
        a = 0.0

    if (
        cross_provider
        and h >= 0.50
        and a >= 0.50
        and morph
        and rederived
    ):
        return (
            "visually_and_morphologically_supported"
        )

    if morph and rederived:
        return (
            "linguistically_plausible_but_"
            "visually_unresolved"
        )

    if morph:
        return (
            "lexically_analyzable_but_"
            "derivationally_unconfirmed"
        )

    return "morphologically_unresolved"


def _tokenize_segmented_sequence(
    text: str,
) -> List[str]:
    if not text:
        return []

    return [
        token
        for token in text.split()
        if token
    ]


def _sequence_morphology(
    sequence: Dict[str, Any],
    *,
    lipi: Any,
    kosha: Any,
    vyakarana: Any,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    tokens = _tokenize_segmented_sequence(
        str(
            sequence.get(
                "vidyut_segmented_readable",
                "",
            )
            or ""
        )
    )

    token_rows: List[Dict[str, Any]] = []

    for index, token_deva in enumerate(
        tokens,
        start=1,
    ):
        slp1 = _to_slp1(
            token_deva,
            lipi,
        )

        if slp1 not in cache:
            cache[slp1] = _analyse_surface(
                slp1,
                kosha=kosha,
                vyakarana=vyakarana,
                lipi=lipi,
            )

        analysis = cache[slp1]

        token_rows.append(
            {
                "token_index": index,
                "token_devanagari": token_deva,
                "token_slp1": slp1,
                "morphologically_analyzable": (
                    analysis[
                        "morphologically_analyzable"
                    ]
                ),
                "exact_surface_rederived": (
                    analysis[
                        "exact_surface_rederived"
                    ]
                ),
                "entry_kinds": (
                    analysis["entry_kinds"]
                ),
                "lemmas_devanagari": (
                    analysis[
                        "lemmas_devanagari"
                    ]
                ),
            }
        )

    total = len(token_rows)

    analyzable = sum(
        bool(
            row[
                "morphologically_analyzable"
            ]
        )
        for row in token_rows
    )

    rederived = sum(
        bool(
            row[
                "exact_surface_rederived"
            ]
        )
        for row in token_rows
    )

    return {
        "provider": sequence.get(
            "provider"
        ),
        "hypothesis_rank": sequence.get(
            "hypothesis_rank"
        ),
        "observed_readable": sequence.get(
            "observed_readable"
        ),
        "vidyut_segmented_readable": sequence.get(
            "vidyut_segmented_readable"
        ),
        "token_count": total,
        "morphologically_analyzable_tokens": analyzable,
        "derivationally_confirmed_tokens": rederived,
        "morphology_coverage": round(
            analyzable
            / float(max(1, total)),
            4,
        ),
        "derivational_coverage": round(
            rederived
            / float(max(1, total)),
            4,
        ),
        "tokens": token_rows,
        "interpretation": (
            "linguistic-coverage evidence only; "
            "not manuscript correctness"
        ),
    }


def run_layer6c_morphology_validation(
    store: ArtifactStore,
    *,
    vidyut_data_root: str = (
        "models/vidyut-0.4.0"
    ),
) -> Dict[str, Any]:
    """
    Stage 6C — conservative morphology and grammar evidence.

    Inputs:
      L6/stage6b_manifest.json

    Guarantees:
      - does not invent replacement Sanskrit;
      - does not treat dictionary membership as visual correctness;
      - uses Vidyut Kosha for structured morphological analyses;
      - uses Vidyut Vyakarana only to verify whether observed forms can be
        generated from returned morphological entries;
      - does not emit final transcription, translation, or T(p).
    """
    run_dir = Path(store.run_dir)

    stage6b = _load_json(
        run_dir
        / "L6"
        / "stage6b_manifest.json"
    )

    lines = stage6b.get("lines")

    if not isinstance(lines, list) or not lines:
        raise RuntimeError(
            "Stage 6C requires Stage 6B line results."
        )

    lipi, kosha, vyakarana = (
        _load_vidyut(
            vidyut_data_root
        )
    )

    cache: Dict[
        str,
        Dict[str, Any],
    ] = {}

    line_results: List[
        Dict[str, Any]
    ] = []

    totals = {
        "candidate_clusters": 0,
        "morphologically_analyzable_clusters": 0,
        "derivationally_confirmed_clusters": 0,
        "visually_and_morphologically_supported_clusters": 0,
        "linguistically_plausible_but_visually_unresolved_clusters": 0,
        "morphologically_unresolved_clusters": 0,
        "sequence_candidates": 0,
    }

    for line in sorted(
        lines,
        key=lambda row: int(
            row.get(
                "reading_order",
                10**9,
            )
        ),
    ):
        line_id = str(
            line.get(
                "line_id",
                "",
            )
            or ""
        )

        H = line.get("H")

        agreement = line.get(
            "cross_provider_agreement"
        )

        validated_clusters: List[
            Dict[str, Any]
        ] = []

        for cluster in line.get(
            "candidate_clusters",
            [],
        ):
            slp1 = str(
                cluster.get(
                    "representative_slp1",
                    "",
                )
                or ""
            )

            if slp1 not in cache:
                cache[slp1] = (
                    _analyse_surface(
                        slp1,
                        kosha=kosha,
                        vyakarana=vyakarana,
                        lipi=lipi,
                    )
                )

            morphology = cache[slp1]

            status = (
                _cluster_validation_status(
                    cluster=cluster,
                    morphology=morphology,
                    H=H,
                    agreement=agreement,
                )
            )

            validated_clusters.append(
                {
                    "cluster_id": cluster.get(
                        "cluster_id"
                    ),
                    "representative_devanagari": (
                        cluster.get(
                            "representative_devanagari"
                        )
                    ),
                    "representative_slp1": slp1,
                    "stage6b_evidence_score": (
                        cluster.get(
                            "evidence_score"
                        )
                    ),
                    "stage6b_evidence_status": (
                        cluster.get(
                            "evidence_status"
                        )
                    ),
                    "cross_provider_support": (
                        cluster.get(
                            "cross_provider_support",
                            False,
                        )
                    ),
                    "morphology": morphology,
                    "stage6c_status": status,
                    "promotion_to_final_text_allowed": False,
                }
            )

            totals[
                "candidate_clusters"
            ] += 1

            if morphology[
                "morphologically_analyzable"
            ]:
                totals[
                    "morphologically_analyzable_clusters"
                ] += 1

            if morphology[
                "exact_surface_rederived"
            ]:
                totals[
                    "derivationally_confirmed_clusters"
                ] += 1

            if status == (
                "visually_and_morphologically_supported"
            ):
                totals[
                    "visually_and_morphologically_supported_clusters"
                ] += 1

            elif status == (
                "linguistically_plausible_but_"
                "visually_unresolved"
            ):
                totals[
                    "linguistically_plausible_but_visually_unresolved_clusters"
                ] += 1

            elif status == (
                "morphologically_unresolved"
            ):
                totals[
                    "morphologically_unresolved_clusters"
                ] += 1

        sequence_rows: List[
            Dict[str, Any]
        ] = []

        for sequence in line.get(
            "provisional_sequence_candidates",
            [],
        ):
            sequence_rows.append(
                _sequence_morphology(
                    sequence,
                    lipi=lipi,
                    kosha=kosha,
                    vyakarana=vyakarana,
                    cache=cache,
                )
            )

        totals[
            "sequence_candidates"
        ] += len(
            sequence_rows
        )

        line_results.append(
            {
                "line_id": line_id,
                "reading_order": line.get(
                    "reading_order"
                ),
                "H": H,
                "cross_provider_agreement": agreement,
                "reconstruction_mode": line.get(
                    "reconstruction_mode"
                ),
                "validated_candidate_clusters": (
                    validated_clusters
                ),
                "sequence_morphology": (
                    sequence_rows
                ),
                "line_conclusion": (
                    "no_candidate_promoted"
                    if not any(
                        item[
                            "stage6c_status"
                        ]
                        == (
                            "visually_and_"
                            "morphologically_supported"
                        )
                        for item in validated_clusters
                    )
                    else (
                        "some_candidates_have_"
                        "independent_visual_and_"
                        "morphological_support"
                    )
                ),
            }
        )

    metrics = {
        "algorithm_version": (
            LAYER6C_VERSION
        ),
        "stage6_substage": (
            "6C_morphology_grammar_validation"
        ),
        "num_lines": len(
            line_results
        ),
        **totals,
        "unique_surface_forms_analysed": (
            len(cache)
        ),
        "generated_unobserved_sanskrit_tokens": 0,
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
            "6C_morphology_grammar_validation"
        ),
        "version": (
            LAYER6C_VERSION
        ),
        "run_id": store.run_id,
        "status": (
            "morphology_evidence_prepared"
        ),
        "vidyut": {
            "data_root": vidyut_data_root,
            "kosha": True,
            "vyakarana": True,
            "vyakarana_nlp_mode": True,
            "is_chandasi": False,
        },
        "safety_contract": {
            "dictionary_membership_is_not_visual_truth": True,
            "morphology_may_not_override_low_H": True,
            "single_provider_linguistic_plausibility_is_not_final_text": True,
            "unobserved_sanskrit_generation_allowed": False,
            "final_transcription_claim_allowed": False,
            "translation_allowed": False,
        },
        "metrics": metrics,
        "lines": line_results,
        "next_action": (
            "run_stage6d_contextual_rag_disambiguation"
        ),
        "audit_note": (
            "Stage 6C enriches observed Stage-6B candidates with structured "
            "Vidyut morphology and optional derivational confirmation. "
            "Linguistic validity is explicitly separated from visual provenance. "
            "No candidate is promoted to final manuscript text in this substage."
        ),
    }

    store.write_json(
        "L6/morphology_evidence.json",
        {
            "version": (
                LAYER6C_VERSION
            ),
            "run_id": store.run_id,
            "lines": line_results,
        },
    )

    store.write_json(
        "L6/stage6c_manifest.json",
        manifest,
    )

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
