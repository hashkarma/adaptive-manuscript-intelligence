from __future__ import annotations

import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.artifact_store import ArtifactStore


LAYER6B_VERSION = "0.2.0-position-aware-candidate-lattice"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 6B input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_weight(rank: int) -> float:
    return {1: 1.0, 2: 0.72, 3: 0.52}.get(rank, max(0.20, 0.52 * (0.82 ** max(0, rank - 3))))


def _clean_spaces(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for ch in "।॥|/\\,;:?!()[]{}\"":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return [t for t in _clean_spaces(text).split() if t]


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if ca == cb else 1)

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def _slp1_similarity(a: str, b: str) -> float:
    """
    Similarity on SLP1 rather than raw Devanagari code points.

    This avoids giving accidental weight to Unicode combining representation
    differences. It is still only a visual/sequence proxy, not linguistic
    correctness.
    """
    a = (a or "").strip()
    b = (b or "").strip()

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    length_ratio = min(len(a), len(b)) / float(
        max(len(a), len(b))
    )

    if length_ratio < 0.50:
        return 0.0

    distance = _levenshtein_distance(
        a,
        b,
    )

    return max(
        0.0,
        1.0
        - distance
        / float(
            max(
                len(a),
                len(b),
            )
        ),
    )


def _analysis_repr(data: Any) -> Tuple[Optional[str], str]:
    if data is None:
        return None, ""
    lemma = None
    for attr in ("lemma", "clean_text", "text"):
        try:
            value = getattr(data, attr)
            if callable(value):
                value = value()
            if isinstance(value, str) and value:
                lemma = value
                break
        except Exception:
            pass
    try:
        rep = repr(data)
    except Exception:
        rep = f"<{type(data).__name__}>"
    return lemma, rep[:600]


def _load_vidyut(data_root: str):
    try:
        from vidyut import lipi
        from vidyut.kosha import Kosha
        from vidyut.cheda import Chedaka
    except Exception as exc:
        raise RuntimeError("Activate venv-stage6; Vidyut import failed.") from exc

    root = Path(data_root)
    kosha_path = root / "kosha"
    cheda_path = root / "cheda"

    if not kosha_path.exists():
        raise FileNotFoundError(f"Missing Vidyut kosha data: {kosha_path}")
    if not cheda_path.exists():
        raise FileNotFoundError(f"Missing Vidyut cheda data: {cheda_path}")

    # IMPORTANT:
    # Kosha expects its own data directory, but Chedaka expects the *Vidyut
    # data root* because it internally needs multiple sibling resources,
    # including:
    #   root/cheda/model.msgpack
    #   root/sandhi/rules.csv
    #   root/kosha/...
    #
    # Passing root/cheda causes:
    #   ValueError: chedaka error, Sandhi(... No such file or directory)
    # because Chedaka then looks for sandhi data beneath the wrong directory.
    return lipi, Kosha(str(kosha_path)), Chedaka(str(root))


def _to_slp1(text: str, lipi: Any) -> str:
    return lipi.transliterate(text, lipi.Scheme.Devanagari, lipi.Scheme.Slp1) if text else ""


def _to_deva(text: str, lipi: Any) -> str:
    return lipi.transliterate(text, lipi.Scheme.Slp1, lipi.Scheme.Devanagari) if text else ""


def _kosha_lookup(kosha: Any, slp1: str) -> Tuple[bool, int]:
    if not slp1:
        return False, 0
    try:
        entries = kosha.get(slp1)
        try:
            n = len(entries)
        except Exception:
            entries = list(entries)
            n = len(entries)
        return n > 0, int(n)
    except Exception:
        return False, 0


def _cheda_evidence(
    *,
    line_id: str,
    provider: str,
    rank: int,
    readable: str,
    lipi: Any,
    chedaka: Any,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    cleaned = _clean_spaces(readable)
    if not cleaned:
        return [], []

    slp1 = _to_slp1(cleaned, lipi)

    try:
        parsed = chedaka.run(slp1)
    except Exception as exc:
        return [], [f"CHEDA_FAILED:{type(exc).__name__}"]

    out: List[Dict[str, Any]] = []
    for i, token in enumerate(parsed, start=1):
        try:
            token_slp1 = str(token.text)
        except Exception:
            token_slp1 = ""

        try:
            data = token.data
        except Exception:
            data = None

        lemma_slp1, rep = _analysis_repr(data)

        out.append(
            {
                "line_id": line_id,
                "provider": provider,
                "hypothesis_rank": rank,
                "token_index": i,
                "token_slp1": token_slp1,
                "token_devanagari": _to_deva(token_slp1, lipi),
                "lemma_slp1": lemma_slp1,
                "lemma_devanagari": _to_deva(lemma_slp1, lipi) if lemma_slp1 else None,
                "analysis_repr": rep,
            }
        )

    return out, []


def _observed_tokens(
    *,
    line_id: str,
    provider: str,
    rank: int,
    readable: str,
    lipi: Any,
    kosha: Any,
    cheda_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cheda_slp1 = {r["token_slp1"] for r in cheda_rows if r.get("token_slp1")}
    out: List[Dict[str, Any]] = []

    hypothesis_tokens = _tokens(readable)
    token_count = len(hypothesis_tokens)

    for i, token in enumerate(hypothesis_tokens, start=1):
        slp1 = _to_slp1(token, lipi)
        exact, count = _kosha_lookup(kosha, slp1)

        normalized_position = (
            (i - 0.5) / float(token_count)
            if token_count
            else 0.0
        )

        out.append(
            {
                "line_id": line_id,
                "provider": provider,
                "hypothesis_rank": rank,
                "token_index": i,
                "token_count": token_count,
                "normalized_position": round(
                    normalized_position,
                    4,
                ),
                "devanagari": token,
                "slp1": slp1,
                "rank_weight": round(_rank_weight(rank), 4),
                "kosha_exact": exact,
                "kosha_entry_count": count,
                # Chedaka can segment noisy strings, so this is segmentation
                # evidence only. It must not be treated as lexical correctness.
                "cheda_supported": slp1 in cheda_slp1,
            }
        )

    return out


def _line_gate(
    H: Optional[float],
    agreement: Optional[float],
) -> float:
    """
    Conservative line-level evidence gate.

    Low H/agreement cannot be magically converted into high-trust evidence by
    lexical plausibility. The floor preserves useful local evidence for later
    Stage-6 reasoning while preventing false "strong" labels.
    """
    try:
        h = float(H)
    except (TypeError, ValueError):
        h = 0.0

    try:
        a = float(agreement)
    except (TypeError, ValueError):
        a = 0.0

    h = min(1.0, max(0.0, h))
    a = min(1.0, max(0.0, a))

    combined = (h * a) ** 0.5

    return 0.55 + 0.45 * combined


def _cluster_tokens(
    observed: List[Dict[str, Any]],
    threshold: float,
    *,
    position_tolerance: float,
    H: Optional[float],
    agreement: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Position-aware clustering of observed HTR tokens.

    v0.1 clustered every similar-looking token anywhere on the line. That could
    incorrectly group unrelated Sanskrit-looking hallucinations such as tokens
    sharing a common -as/-ās ending.

    v0.2 requires BOTH:
      1. SLP1 edit similarity above threshold, and
      2. roughly compatible normalized token position within the line.

    No new Sanskrit spelling is generated.
    """
    raw_clusters: List[List[Dict[str, Any]]] = []

    for token in observed:
        best_idx = None
        best_score = 0.0

        for idx, cluster in enumerate(raw_clusters):
            rep = max(
                cluster,
                key=lambda x: (
                    x["rank_weight"],
                    x["kosha_exact"],
                    -x["hypothesis_rank"],
                ),
            )

            position_gap = abs(
                float(token.get("normalized_position", 0.0))
                - float(rep.get("normalized_position", 0.0))
            )

            if position_gap > position_tolerance:
                continue

            score = _slp1_similarity(
                token["slp1"],
                rep["slp1"],
            )

            if score >= threshold and score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            raw_clusters.append([token])
        else:
            raw_clusters[best_idx].append(token)

    clusters: List[Dict[str, Any]] = []
    line_gate = _line_gate(
        H,
        agreement,
    )

    for index, members in enumerate(raw_clusters, start=1):
        providers = {
            m["provider"]
            for m in members
        }

        # Multiple N-best ranks from one provider are correlated evidence, not
        # independent recognizers. Track them, but cap their contribution.
        provider_rank_sets = {
            provider: {
                m["hypothesis_rank"]
                for m in members
                if m["provider"] == provider
            }
            for provider in providers
        }

        independent_provider_count = len(providers)
        cross_provider = independent_provider_count > 1

        rep = max(
            members,
            key=lambda m: (
                0.55 * m["rank_weight"]
                + 0.25 * float(m["kosha_exact"])
                + 0.05 * float(m["cheda_supported"])
                + 0.15 * float(cross_provider)
            ),
        )

        lexical_ratio = (
            sum(bool(m["kosha_exact"]) for m in members)
            / float(len(members))
        )

        cheda_ratio = (
            sum(bool(m["cheda_supported"]) for m in members)
            / float(len(members))
        )

        best_rank_support = max(
            m["rank_weight"]
            for m in members
        )

        # Same-provider rank recurrence is only a modest stability signal.
        same_provider_stability = max(
            min(1.0, len(ranks) / 3.0)
            for ranks in provider_rank_sets.values()
        )

        # Independent visual corroboration is the dominant Stage-6B evidence.
        local_evidence_score = (
            0.46 * float(cross_provider)
            + 0.30 * best_rank_support
            + 0.10 * same_provider_stability
            + 0.09 * lexical_ratio
            + 0.05 * cheda_ratio
        )

        adjusted_score = (
            local_evidence_score
            * line_gate
        )

        # "Strong" requires independent provider support. A valid Sanskrit word
        # repeatedly hallucinated by one model is not strong manuscript evidence.
        if (
            cross_provider
            and adjusted_score >= 0.72
        ):
            status = "strong_observed_candidate"
        elif adjusted_score >= 0.46:
            status = "moderate_observed_candidate"
        else:
            status = "weak_observed_candidate"

        normalized_positions = [
            float(
                m.get(
                    "normalized_position",
                    0.0,
                )
            )
            for m in members
        ]

        clusters.append(
            {
                "cluster_id": f"cluster_{index:03d}",
                "representative_devanagari": rep["devanagari"],
                "representative_slp1": rep["slp1"],
                "representative_is_observed": True,
                "provider_count": independent_provider_count,
                "hypothesis_count": sum(
                    len(ranks)
                    for ranks in provider_rank_sets.values()
                ),
                "best_rank": min(
                    m["hypothesis_rank"]
                    for m in members
                ),
                "mean_normalized_position": round(
                    sum(normalized_positions)
                    / float(len(normalized_positions)),
                    4,
                ),
                "position_span": round(
                    max(normalized_positions)
                    - min(normalized_positions),
                    4,
                ),
                "lexical_support_ratio": round(
                    lexical_ratio,
                    4,
                ),
                "cheda_support_ratio": round(
                    cheda_ratio,
                    4,
                ),
                "cross_provider_support": cross_provider,
                "same_provider_nbest_stability": round(
                    same_provider_stability,
                    4,
                ),
                "line_evidence_gate": round(
                    line_gate,
                    4,
                ),
                "local_evidence_score": round(
                    local_evidence_score,
                    4,
                ),
                "evidence_score": round(
                    adjusted_score,
                    4,
                ),
                "evidence_status": status,
                "members": members,
            }
        )

    clusters.sort(
        key=lambda c: (
            c["evidence_score"],
            c["cross_provider_support"],
            -c["best_rank"],
        ),
        reverse=True,
    )

    return clusters


def run_layer6b_candidate_reconstruction(
    store: ArtifactStore,
    *,
    vidyut_data_root: str = "models/vidyut-0.4.0",
    similarity_threshold: float = 0.72,
    position_tolerance: float = 0.14,
) -> Dict[str, Any]:
    """
    Stage 6B — conservative Vidyut Viccheda + observed-token candidate lattice.

    No unobserved Sanskrit token is generated here.
    No final transcription, T(p), translation, or scholar routing is produced.
    """
    run_dir = Path(store.run_dir)
    stage6a = _load_json(run_dir / "L6" / "htr_input_table.json")
    rows = stage6a.get("rows")

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Stage 6A rows are empty.")

    lipi, kosha, chedaka = _load_vidyut(vidyut_data_root)

    line_results: List[Dict[str, Any]] = []
    totals = {
        "observed_htr_tokens": 0,
        "vidyut_cheda_tokens": 0,
        "candidate_clusters": 0,
        "strong_observed_candidate_clusters": 0,
        "cross_provider_candidate_clusters": 0,
    }

    for row in sorted(rows, key=lambda r: int(r.get("reading_order", 10**9))):
        line_id = str(row.get("line_id", "") or "")
        if not line_id:
            continue

        observed: List[Dict[str, Any]] = []
        cheda_all: List[Dict[str, Any]] = []
        sequences: List[Dict[str, Any]] = []
        notes: List[str] = []

        for provider, key in (
            ("provider_a", "provider_a_hypotheses"),
            ("provider_b", "provider_b_hypotheses"),
        ):
            hypotheses = row.get(key, [])
            if not isinstance(hypotheses, list):
                hypotheses = []

            for hyp in hypotheses:
                if not isinstance(hyp, dict):
                    continue

                rank = int(hyp.get("rank", 1))
                readable = str(hyp.get("readable_devanagari", "") or "")

                cheda_rows, cheda_notes = _cheda_evidence(
                    line_id=line_id,
                    provider=provider,
                    rank=rank,
                    readable=readable,
                    lipi=lipi,
                    chedaka=chedaka,
                )

                cheda_all.extend(cheda_rows)
                notes.extend(f"{provider}:rank_{rank}:{n}" for n in cheda_notes)

                segmented = " ".join(
                    r["token_devanagari"]
                    for r in cheda_rows
                    if r.get("token_devanagari")
                )

                sequences.append(
                    {
                        "provider": provider,
                        "hypothesis_rank": rank,
                        "observed_readable": readable,
                        "vidyut_segmented_readable": segmented,
                        "segment_count": len(cheda_rows),
                        "status": (
                            "segmented_observed_hypothesis"
                            if cheda_rows
                            else "unsegmented_observed_hypothesis"
                        ),
                    }
                )

                observed.extend(
                    _observed_tokens(
                        line_id=line_id,
                        provider=provider,
                        rank=rank,
                        readable=readable,
                        lipi=lipi,
                        kosha=kosha,
                        cheda_rows=cheda_rows,
                    )
                )

        clusters = _cluster_tokens(
            observed,
            similarity_threshold,
            position_tolerance=position_tolerance,
            H=row.get("htr_readiness_H"),
            agreement=row.get("cross_provider_agreement"),
        )

        if not any(c["cross_provider_support"] for c in clusters):
            notes.append("NO_FUZZY_CROSS_PROVIDER_TOKEN_SUPPORT")

        line_result = {
            "line_id": line_id,
            "reading_order": int(row.get("reading_order", len(line_results) + 1)),
            "H": row.get("htr_readiness_H"),
            "cross_provider_agreement": row.get("cross_provider_agreement"),
            "reconstruction_mode": row.get(
                "reconstruction_mode",
                "deep_semantic_reconstruction",
            ),
            "observed_tokens": observed,
            "cheda_evidence": cheda_all,
            "candidate_clusters": clusters,
            "provisional_sequence_candidates": sequences,
            "notes": sorted(set(notes)),
        }

        line_results.append(line_result)

        totals["observed_htr_tokens"] += len(observed)
        totals["vidyut_cheda_tokens"] += len(cheda_all)
        totals["candidate_clusters"] += len(clusters)
        totals["strong_observed_candidate_clusters"] += sum(
            c["evidence_status"] == "strong_observed_candidate"
            for c in clusters
        )
        totals["cross_provider_candidate_clusters"] += sum(
            c["cross_provider_support"]
            for c in clusters
        )

    metrics = {
        "algorithm_version": LAYER6B_VERSION,
        "stage6_substage": "6B_viccheda_candidate_lattice",
        "num_lines": len(line_results),
        **totals,
        "similarity_threshold": similarity_threshold,
        "position_tolerance": position_tolerance,
        "candidate_clustering": (
            "SLP1 edit similarity + normalized line-position compatibility"
        ),
        "single_provider_candidate_can_be_strong": False,
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
        "substage": "6B_viccheda_candidate_lattice",
        "version": LAYER6B_VERSION,
        "run_id": store.run_id,
        "status": "candidate_evidence_prepared",
        "vidyut": {
            "data_root": vidyut_data_root,
            "kosha_path": str(Path(vidyut_data_root) / "kosha"),
            "cheda_path": str(Path(vidyut_data_root) / "cheda"),
            "input_script": "SLP1",
            "pipeline_script": "Devanagari",
        },
        "safety_contract": {
            "cluster_representatives_must_be_observed_htr_tokens": True,
            "vidyut_is_evidence_not_ground_truth": True,
            "unobserved_sanskrit_generation_allowed": False,
            "final_transcription_claim_allowed": False,
            "translation_allowed": False,
        },
        "metrics": metrics,
        "lines": line_results,
        "next_action": "run_stage6c_morphology_and_grammar_validation",
        "audit_note": (
            "Stage 6B clusters only position-compatible observed HTR tokens "
            "using SLP1 edit similarity, queries Vidyut Kosha, and segments observed "
            "hypotheses with Vidyut Chedaka. Same-provider N-best recurrence is "
            "treated as correlated evidence, lexical validity cannot independently "
            "create a strong candidate, and low H/agreement conservatively gates "
            "candidate strength. No corrected Sanskrit is invented."
        ),
    }

    store.write_json(
        "L6/candidate_lattice.json",
        {
            "version": LAYER6B_VERSION,
            "run_id": store.run_id,
            "lines": [
                {
                    "line_id": line["line_id"],
                    "reading_order": line["reading_order"],
                    "candidate_clusters": line["candidate_clusters"],
                }
                for line in line_results
            ],
        },
    )

    store.write_json(
        "L6/viccheda_evidence.json",
        {
            "version": LAYER6B_VERSION,
            "run_id": store.run_id,
            "lines": [
                {
                    "line_id": line["line_id"],
                    "reading_order": line["reading_order"],
                    "provisional_sequence_candidates": line[
                        "provisional_sequence_candidates"
                    ],
                    "cheda_evidence": line["cheda_evidence"],
                }
                for line in line_results
            ],
        },
    )

    store.write_json("L6/stage6b_manifest.json", manifest)

    return {
        "lines": line_results,
        "metrics": metrics,
        "manifest": manifest,
    }
