from __future__ import annotations

import json
import math
import os
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from core.artifact_store import ArtifactStore
from layers.htr_providers import (
    DEFAULT_TROCR_IAST_MODEL_ID,
    ProviderHypothesis,
    create_htr_provider,
)

LAYER5_VERSION = "0.4.2-iast-edge-normalization"
DEFAULT_PROVIDER = "trocr_iast_baseline"
DEFAULT_MODEL_ID = DEFAULT_TROCR_IAST_MODEL_ID

# Canonically equivalent, selective preparation rules for model-emitted IAST.
#
# IMPORTANT:
# We deliberately do NOT run global NFD normalization because our controlled
# test showed that indic-transliteration then leaves some decomposed IAST
# marks unresolved (e.g. macrons/dots in otherwise valid ā/ī/ṣ/ṭ/ṇ forms).
#
# U+0155 LATIN SMALL LETTER R WITH ACUTE canonically decomposes to:
#   U+0072 LATIN SMALL LETTER R
#   U+0301 COMBINING ACUTE ACCENT
#
# This rule is Unicode-equivalent preparation, not a linguistic correction.
SELECTIVE_IAST_CANONICAL_DECOMPOSITION = {
    "ŕ": "r\u0301",
    "Ŕ": "R\u0301",
}

# Observed model-output alias:
#   l + U+0325 COMBINING RING BELOW
# is used by the OCR model for vocalic l. indic-transliteration does not
# consume that sequence in IAST mode, while it correctly handles the IAST
# vocalic-l character ḷ.
#
# IMPORTANT:
# This is NOT a Unicode-canonical decomposition. It is an explicit,
# experimentally validated transliteration-alias normalization.
SELECTIVE_IAST_ALIAS_NORMALIZATION = {
    "l\u0325": "ḷ",
}

# Post-transliteration repair for a combining candrabindu that survives the
# IAST -> Devanagari conversion. A correctly consumed candrabindu is already
# converted by the library and therefore is never touched by this rule.
POST_TRANSLITERATION_RESIDUAL_REPAIR = {
    "\u0310": "\u0901",
}


@dataclass
class HTRHypothesis:
    rank: int
    raw_text: str
    raw_script: str
    raw_iast: Optional[str]
    transliteration_input: Optional[str]
    devanagari_text: Optional[str]
    normalization_actions: List[str]
    relative_score: Optional[float]
    sequence_score: Optional[float]


@dataclass
class HTRQuality:
    hypothesis_entropy: Optional[float]
    top1_relative_score: Optional[float]
    top2_margin: Optional[float]
    decoder_stability: Optional[float]

    script_purity: Optional[float]
    latin_residual_ratio: Optional[float]
    unexpected_character_ratio: Optional[float]

    transliteration_valid: bool
    unicode_valid: bool
    unexpected_characters: List[str]

    token_count: int
    unique_token_count: int
    token_repetition_ratio: Optional[float]
    longest_consecutive_repeat: int
    repeated_tokens: List[str]

    hypothesis_length_spread: Optional[float]

    image_width: int
    image_height: int
    output_char_count: int
    chars_per_100px: Optional[float]
    page_length_robust_z: Optional[float]
    sequence_length_anomaly_score: Optional[float]

    warnings: List[str]


@dataclass
class HTRLineResult:
    line_id: str
    reading_order: int
    source_crop: str

    status: str

    raw_text: str
    raw_script: str
    raw_iast: Optional[str]
    transliteration_input: Optional[str]
    devanagari_text: Optional[str]
    normalization_actions: List[str]

    hypotheses: List[HTRHypothesis]
    quality: HTRQuality

    review_required: bool
    runtime_ms: float
    device_used: Optional[str]

    error: Optional[str] = None


@dataclass
class Layer5Output:
    lines: List[HTRLineResult]
    metrics: Dict[str, Any]
    manifest: Dict[str, Any]


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """
    Standard provider-contract normalization:
      - NFC Unicode normalization
      - trim leading/trailing whitespace
      - collapse runs of whitespace

    The provider's recognized lexical content is otherwise preserved.
    """
    return " ".join(
        unicodedata.normalize("NFC", text or "")
        .strip()
        .split()
    )


def _prepare_iast_for_transliteration(
    text: str,
) -> Tuple[str, List[str]]:
    """
    Prepare IAST for indic-transliteration without changing stored raw IAST.

    Policy:
      1. Keep the entire string in NFC.
      2. Apply only explicitly validated canonical-equivalence rules.
      3. Apply only explicitly validated model-output transliteration aliases.
      4. Never run global NFD.
    """
    text = _normalise_text(text)
    actions: List[str] = []

    prepared_chars: List[str] = []

    for ch in text:
        replacement = SELECTIVE_IAST_CANONICAL_DECOMPOSITION.get(ch)

        if replacement is not None:
            prepared_chars.append(replacement)
            actions.append(
                f"canonical_decomposition:U+{ord(ch):04X}"
            )
        else:
            prepared_chars.append(ch)

    prepared = "".join(prepared_chars)

    for source, target in SELECTIVE_IAST_ALIAS_NORMALIZATION.items():
        if source in prepared:
            prepared = prepared.replace(source, target)
            actions.append(
                "transliteration_alias:"
                + "+".join(f"U+{ord(ch):04X}" for ch in source)
                + f"->U+{ord(target):04X}"
            )

    return prepared, list(dict.fromkeys(actions))


def _repair_devanagari_residuals(
    text: str,
) -> Tuple[str, List[str]]:
    """
    Repair only explicitly validated combining marks that survive conversion.

    Current rule:
      U+0310 COMBINING CANDRABINDU -> U+0901 DEVANAGARI SIGN CANDRABINDU

    This is intentionally post-transliteration so cases already handled by
    indic-transliteration are left unchanged.
    """
    actions: List[str] = []
    repaired = text

    for source, target in POST_TRANSLITERATION_RESIDUAL_REPAIR.items():
        if source in repaired:
            repaired = repaired.replace(source, target)
            actions.append(
                f"post_transliteration_residual:"
                f"U+{ord(source):04X}->U+{ord(target):04X}"
            )

    return (
        unicodedata.normalize("NFC", repaired),
        list(dict.fromkeys(actions)),
    )


def _iast_to_devanagari(
    text: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Return:
      (transliteration_input, devanagari_text, normalization_actions)

    The original model output is preserved separately as raw_iast.
    """
    raw_iast = _normalise_text(text)

    if not raw_iast:
        return "", "", []

    prepared, actions = _prepare_iast_for_transliteration(raw_iast)

    try:
        from indic_transliteration.sanscript import (
            DEVANAGARI,
            IAST,
            transliterate,
        )

        devanagari = transliterate(
            prepared,
            IAST,
            DEVANAGARI,
        )

        repaired, repair_actions = _repair_devanagari_residuals(
            devanagari
        )

        actions.extend(repair_actions)

        return (
            prepared,
            _normalise_text(repaired),
            list(dict.fromkeys(actions)),
        )

    except Exception:
        return prepared, None, list(dict.fromkeys(actions))


def _provider_to_hypotheses(
    rows: Sequence[ProviderHypothesis],
    output_script: str,
) -> List[HTRHypothesis]:

    output_script = output_script.strip().lower()

    result: List[HTRHypothesis] = []

    for row in rows:
        raw = _normalise_text(row.raw_text)

        if output_script == "iast":
            raw_iast = raw
            (
                transliteration_input,
                devanagari,
                normalization_actions,
            ) = _iast_to_devanagari(raw)

        elif output_script == "devanagari":
            raw_iast = None
            transliteration_input = None
            devanagari = raw
            normalization_actions = []

        else:
            raw_iast = None
            transliteration_input = None
            devanagari = None
            normalization_actions = []

        result.append(
            HTRHypothesis(
                rank=row.rank,
                raw_text=raw,
                raw_script=output_script,
                raw_iast=raw_iast,
                transliteration_input=transliteration_input,
                devanagari_text=devanagari,
                normalization_actions=normalization_actions,
                relative_score=row.relative_score,
                sequence_score=row.sequence_score,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Decoder diagnostics
# ---------------------------------------------------------------------------

def _normalised_entropy(
    probabilities: Sequence[float],
) -> Optional[float]:

    probs = [
        float(p)
        for p in probabilities
        if p is not None and p > 0.0
    ]

    if len(probs) < 2:
        return None

    total = sum(probs)

    if total <= 0:
        return None

    probs = [
        p / total
        for p in probs
    ]

    entropy = -sum(
        p * math.log(p)
        for p in probs
    )

    max_entropy = math.log(
        len(probs)
    )

    if max_entropy <= 0:
        return 0.0

    return round(
        max(
            0.0,
            min(
                1.0,
                entropy / max_entropy,
            ),
        ),
        4,
    )


def _decoder_metrics(
    relative_scores: Sequence[float],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:

    scores = sorted(
        [
            float(v)
            for v in relative_scores
            if v is not None
        ],
        reverse=True,
    )

    if not scores:
        return (
            None,
            None,
            None,
            None,
        )

    top1 = scores[0]

    if len(scores) < 2:
        return (
            None,
            round(top1, 4),
            None,
            None,
        )

    margin = max(
        0.0,
        top1 - scores[1],
    )

    entropy = _normalised_entropy(
        scores
    )

    stability = None

    if entropy is not None:
        stability = round(
            max(
                0.0,
                min(
                    1.0,
                    0.45 * (1.0 - entropy)
                    + 0.35 * top1
                    + 0.20 * margin,
                ),
            ),
            4,
        )

    return (
        entropy,
        round(top1, 4),
        round(margin, 4),
        stability,
    )


# ---------------------------------------------------------------------------
# Devanagari / Unicode diagnostics
# ---------------------------------------------------------------------------

def _is_expected_devanagari_char(
    ch: str,
) -> bool:

    cp = ord(ch)

    return (
        0x0900 <= cp <= 0x097F
        or 0x1CD0 <= cp <= 0x1CFF
        or 0xA8E0 <= cp <= 0xA8FF
        or 0x11B00 <= cp <= 0x11B5F
    )


def _is_neutral_char(
    ch: str,
) -> bool:

    if ch.isspace():
        return True

    category = unicodedata.category(ch)

    return (
        category.startswith("P")
        or category == "Nd"
    )


def _character_diagnostics(
    devanagari_text: Optional[str],
) -> Dict[str, Any]:

    if devanagari_text is None:
        return {
            "script_purity": 0.0,
            "latin_residual_ratio": 1.0,
            "unexpected_character_ratio": 1.0,
            "unicode_valid": False,
            "unexpected_characters": [],
        }

    text = unicodedata.normalize(
        "NFC",
        devanagari_text,
    )

    expected = 0
    unexpected = 0
    latin = 0
    invalid_unicode = 0

    unexpected_chars: List[str] = []

    for ch in text:

        cp = ord(ch)
        category = unicodedata.category(ch)

        if (
            0xD800 <= cp <= 0xDFFF
            or (
                category.startswith("C")
                and not ch.isspace()
            )
        ):
            invalid_unicode += 1
            unexpected += 1
            unexpected_chars.append(ch)
            continue

        if _is_expected_devanagari_char(ch):
            expected += 1
            continue

        if _is_neutral_char(ch):
            continue

        unexpected += 1

        if "LATIN" in unicodedata.name(
            ch,
            "",
        ):
            latin += 1

        unexpected_chars.append(ch)

    content_total = (
        expected
        + unexpected
    )

    if content_total == 0:
        script_purity = 0.0
        latin_ratio = 0.0
        unexpected_ratio = 0.0

    else:
        script_purity = (
            expected
            / content_total
        )

        latin_ratio = (
            latin
            / content_total
        )

        unexpected_ratio = (
            unexpected
            / content_total
        )

    seen = set()
    formatted: List[str] = []

    for ch in unexpected_chars:

        cp = ord(ch)

        if cp in seen:
            continue

        seen.add(cp)

        formatted.append(
            f"{ch} "
            f"[U+{cp:04X} "
            f"{unicodedata.name(ch, 'UNKNOWN')}]"
        )

    return {
        "script_purity": round(
            script_purity,
            4,
        ),
        "latin_residual_ratio": round(
            latin_ratio,
            4,
        ),
        "unexpected_character_ratio": round(
            unexpected_ratio,
            4,
        ),
        "unicode_valid": (
            invalid_unicode == 0
        ),
        "unexpected_characters": (
            formatted
        ),
    }


# ---------------------------------------------------------------------------
# Sequence diagnostics
# ---------------------------------------------------------------------------

def _lexical_tokens(
    text: str,
) -> List[str]:

    text = unicodedata.normalize(
        "NFC",
        text or "",
    )

    tokens: List[str] = []
    current: List[str] = []

    for ch in text:

        category = unicodedata.category(ch)

        if (
            category.startswith("L")
            or category.startswith("M")
        ):
            current.append(
                ch.casefold()
            )

        elif current:
            tokens.append(
                "".join(current)
            )
            current = []

    if current:
        tokens.append(
            "".join(current)
        )

    return [
        token
        for token in tokens
        if token
    ]


def _repetition_diagnostics(
    text: str,
) -> Dict[str, Any]:

    tokens = _lexical_tokens(
        text
    )

    if not tokens:
        return {
            "token_count": 0,
            "unique_token_count": 0,
            "token_repetition_ratio": None,
            "longest_consecutive_repeat": 0,
            "repeated_tokens": [],
        }

    counts: Dict[str, int] = {}

    for token in tokens:
        counts[token] = (
            counts.get(token, 0)
            + 1
        )

    unique_count = len(
        counts
    )

    repetition_ratio = max(
        0.0,
        1.0
        - unique_count / len(tokens),
    )

    longest = 1
    current = 1

    for index in range(
        1,
        len(tokens),
    ):
        if (
            tokens[index]
            == tokens[index - 1]
        ):
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    repeated = [
        f"{token}×{count}"
        for token, count in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
        if count >= 2
    ]

    return {
        "token_count": (
            len(tokens)
        ),
        "unique_token_count": (
            unique_count
        ),
        "token_repetition_ratio": round(
            repetition_ratio,
            4,
        ),
        "longest_consecutive_repeat": (
            longest
        ),
        "repeated_tokens": (
            repeated[:10]
        ),
    }


def _hypothesis_length_spread(
    hypotheses: Sequence[HTRHypothesis],
) -> Optional[float]:

    lengths = [
        len(
            "".join(
                hypothesis.raw_text.split()
            )
        )
        for hypothesis
        in hypotheses
    ]

    if len(lengths) < 2:
        return None

    median_length = statistics.median(
        lengths
    )

    if median_length <= 0:
        return None

    return round(
        (
            max(lengths)
            - min(lengths)
        )
        / median_length,
        4,
    )


# ---------------------------------------------------------------------------
# Shared quality evaluator
# ---------------------------------------------------------------------------

def _evaluate_htr_quality(
    raw_text: str,
    raw_script: str,
    devanagari_text: Optional[str],
    hypotheses: Sequence[HTRHypothesis],
    *,
    image_width: int,
    image_height: int,
) -> HTRQuality:

    relative_scores = [
        float(
            hypothesis.relative_score
        )
        for hypothesis
        in hypotheses
        if hypothesis.relative_score
        is not None
    ]

    (
        entropy,
        top1,
        margin,
        stability,
    ) = _decoder_metrics(
        relative_scores
    )

    chars = _character_diagnostics(
        devanagari_text
    )

    repetition = _repetition_diagnostics(
        raw_text
    )

    length_spread = (
        _hypothesis_length_spread(
            hypotheses
        )
    )

    transliteration_valid = bool(
        devanagari_text is not None
        and raw_text.strip()
        and chars["unicode_valid"]
        and chars["latin_residual_ratio"] == 0.0
        and chars["unexpected_character_ratio"] == 0.0
    )

    output_char_count = len(
        "".join(
            raw_text.split()
        )
    )

    chars_per_100px = (
        None
        if image_width <= 0
        else round(
            (
                output_char_count
                / image_width
            )
            * 100.0,
            4,
        )
    )

    warnings: List[str] = []

    if not raw_text.strip():
        warnings.append(
            "EMPTY_HTR_OUTPUT"
        )

    if entropy is None:
        warnings.append(
            "DECODER_AMBIGUITY_UNAVAILABLE"
        )

    elif entropy >= 0.90:
        warnings.append(
            "HIGH_DECODER_AMBIGUITY"
        )

    if stability is None:
        warnings.append(
            "DECODER_STABILITY_UNAVAILABLE"
        )

    elif stability < 0.35:
        warnings.append(
            "LOW_DECODER_STABILITY"
        )

    if (
        chars["script_purity"]
        < 0.98
    ):
        warnings.append(
            "LOW_DEVANAGARI_SCRIPT_PURITY"
        )

    if (
        chars["latin_residual_ratio"]
        > 0.0
    ):
        warnings.append(
            "LATIN_RESIDUAL_IN_DEVANAGARI_OUTPUT"
        )

    if (
        chars[
            "unexpected_character_ratio"
        ]
        > 0.0
    ):
        warnings.append(
            "UNEXPECTED_CHARACTERS_IN_DEVANAGARI_OUTPUT"
        )

    if not chars["unicode_valid"]:
        warnings.append(
            "INVALID_UNICODE_CONTENT"
        )

    if not transliteration_valid:

        if raw_script == "iast":
            warnings.append(
                "TRANSLITERATION_VALIDATION_FAILED"
            )

        else:
            warnings.append(
                "OUTPUT_SCRIPT_VALIDATION_FAILED"
            )

    if (
        repetition["token_count"] >= 8
        and repetition[
            "token_repetition_ratio"
        ]
        is not None
        and repetition[
            "token_repetition_ratio"
        ]
        >= 0.35
    ):
        warnings.append(
            "POSSIBLE_REPETITIVE_GENERATION"
        )

    if (
        repetition[
            "longest_consecutive_repeat"
        ]
        >= 3
    ):
        warnings.append(
            "CONSECUTIVE_TOKEN_REPETITION"
        )

    if (
        length_spread is not None
        and length_spread >= 0.75
    ):
        warnings.append(
            "N_BEST_LENGTH_DIVERGENCE"
        )

    return HTRQuality(
        hypothesis_entropy=entropy,
        top1_relative_score=top1,
        top2_margin=margin,
        decoder_stability=stability,

        script_purity=(
            chars["script_purity"]
        ),
        latin_residual_ratio=(
            chars[
                "latin_residual_ratio"
            ]
        ),
        unexpected_character_ratio=(
            chars[
                "unexpected_character_ratio"
            ]
        ),

        transliteration_valid=(
            transliteration_valid
        ),
        unicode_valid=(
            chars["unicode_valid"]
        ),
        unexpected_characters=(
            chars[
                "unexpected_characters"
            ]
        ),

        token_count=(
            repetition["token_count"]
        ),
        unique_token_count=(
            repetition[
                "unique_token_count"
            ]
        ),
        token_repetition_ratio=(
            repetition[
                "token_repetition_ratio"
            ]
        ),
        longest_consecutive_repeat=(
            repetition[
                "longest_consecutive_repeat"
            ]
        ),
        repeated_tokens=(
            repetition[
                "repeated_tokens"
            ]
        ),

        hypothesis_length_spread=(
            length_spread
        ),

        image_width=image_width,
        image_height=image_height,
        output_char_count=(
            output_char_count
        ),
        chars_per_100px=(
            chars_per_100px
        ),
        page_length_robust_z=None,
        sequence_length_anomaly_score=None,

        warnings=list(
            dict.fromkeys(
                warnings
            )
        ),
    )


# ---------------------------------------------------------------------------
# Page-relative sequence-length diagnostics
# ---------------------------------------------------------------------------

def _apply_page_length_diagnostics(
    results: Sequence[HTRLineResult],
) -> None:

    successful = [
        result
        for result in results
        if (
            result.status == "ok"
            and result.quality.chars_per_100px
            is not None
        )
    ]

    # Do not derive page-relative statistics from 1-line or 3-line smoke tests.
    if len(successful) < 7:
        return

    densities = [
        float(
            result.quality.chars_per_100px
        )
        for result in successful
    ]

    median_density = statistics.median(
        densities
    )

    mad = statistics.median(
        [
            abs(
                value
                - median_density
            )
            for value in densities
        ]
    )

    if mad <= 1e-9:

        for result in successful:
            result.quality.page_length_robust_z = (
                0.0
            )

            result.quality.sequence_length_anomaly_score = (
                0.0
            )

        return

    for result in successful:

        density = float(
            result.quality.chars_per_100px
        )

        robust_z = (
            0.6745
            * abs(
                density
                - median_density
            )
            / mad
        )

        anomaly_score = min(
            1.0,
            robust_z / 6.0,
        )

        result.quality.page_length_robust_z = round(
            robust_z,
            4,
        )

        result.quality.sequence_length_anomaly_score = round(
            anomaly_score,
            4,
        )

        if robust_z >= 3.5:
            result.quality.warnings.append(
                "PAGE_RELATIVE_SEQUENCE_LENGTH_ANOMALY"
            )

        result.quality.warnings = list(
            dict.fromkeys(
                result.quality.warnings
            )
        )

        result.review_required = bool(
            result.quality.warnings
        )


# ---------------------------------------------------------------------------
# Stage 4 manifest / utility helpers
# ---------------------------------------------------------------------------

def _load_stage4_manifest(
    store: ArtifactStore,
) -> Dict[str, Any]:

    path = os.path.join(
        store.run_dir,
        "L4",
        "line_manifest.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 4 manifest is missing: "
            f"{path}. "
            "Run and validate Stage 4 "
            "before Stage 5."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    lines = manifest.get(
        "lines"
    )

    if (
        not isinstance(lines, list)
        or not lines
    ):
        raise ValueError(
            "Stage 4 manifest contains "
            "no line records."
        )

    return manifest


def _safe_mean(
    values: Sequence[Optional[float]],
) -> Optional[float]:

    valid = [
        float(value)
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return round(
        sum(valid)
        / len(valid),
        4,
    )


def _empty_quality(
    warning: str,
) -> HTRQuality:

    return HTRQuality(
        hypothesis_entropy=None,
        top1_relative_score=None,
        top2_margin=None,
        decoder_stability=None,

        script_purity=None,
        latin_residual_ratio=None,
        unexpected_character_ratio=None,

        transliteration_valid=False,
        unicode_valid=False,
        unexpected_characters=[],

        token_count=0,
        unique_token_count=0,
        token_repetition_ratio=None,
        longest_consecutive_repeat=0,
        repeated_tokens=[],

        hypothesis_length_spread=None,

        image_width=0,
        image_height=0,
        output_char_count=0,
        chars_per_100px=None,
        page_length_robust_z=None,
        sequence_length_anomaly_score=None,

        warnings=[warning],
    )


# ---------------------------------------------------------------------------
# Main Stage 5 entry point
# ---------------------------------------------------------------------------

def run_layer5_htr(
    store: ArtifactStore,
    *,
    provider_name: str = DEFAULT_PROVIDER,
    model_id: Optional[str] = None,
    device: str = "auto",
    num_beams: int = 4,
    n_best: int = 3,
    max_output_length: int = 192,
    max_lines: Optional[int] = None,
) -> Layer5Output:

    stage4_manifest = (
        _load_stage4_manifest(
            store
        )
    )

    stage4_lines = sorted(
        stage4_manifest["lines"],
        key=lambda row: int(
            row.get(
                "reading_order",
                0,
            )
        ),
    )

    if max_lines is not None:
        stage4_lines = stage4_lines[
            : max(
                0,
                int(max_lines),
            )
        ]

    provider = create_htr_provider(
        provider_name,
        model_id=model_id,
        device=device,
    )

    results: List[
        HTRLineResult
    ] = []

    for row in stage4_lines:

        line_id = str(
            row["line_id"]
        )

        reading_order = int(
            row.get(
                "reading_order",
                len(results) + 1,
            )
        )

        crop_rel_path = str(
            row["crop_rel_path"]
        )

        crop_path = os.path.join(
            store.run_dir,
            crop_rel_path,
        )

        if not os.path.exists(
            crop_path
        ):
            results.append(
                HTRLineResult(
                    line_id=line_id,
                    reading_order=reading_order,
                    source_crop=crop_rel_path,

                    status="error",

                    raw_text="",
                    raw_script=(
                        provider.output_script
                    ),
                    raw_iast=None,
                    transliteration_input=None,
                    devanagari_text=None,
                    normalization_actions=[],

                    hypotheses=[],
                    quality=_empty_quality(
                        "MISSING_STAGE4_LINE_CROP"
                    ),

                    review_required=True,
                    runtime_ms=0.0,
                    device_used=None,

                    error=(
                        "Missing Stage 4 crop: "
                        f"{crop_path}"
                    ),
                )
            )

            continue

        try:
            image = (
                Image.open(
                    crop_path
                )
                .convert(
                    "RGB"
                )
            )

            width, height = (
                image.size
            )

            provider_result = (
                provider.recognize(
                    image,
                    num_beams=(
                        num_beams
                    ),
                    n_best=(
                        n_best
                    ),
                    max_output_length=(
                        max_output_length
                    ),
                )
            )

            hypotheses = (
                _provider_to_hypotheses(
                    provider_result.hypotheses,
                    provider.output_script,
                )
            )

            if not hypotheses:
                raise RuntimeError(
                    "The HTR provider "
                    "returned no hypotheses."
                )

            top = hypotheses[0]

            quality = (
                _evaluate_htr_quality(
                    top.raw_text,
                    top.raw_script,
                    top.devanagari_text,
                    hypotheses,
                    image_width=width,
                    image_height=height,
                )
            )

            result = HTRLineResult(
                line_id=line_id,
                reading_order=reading_order,
                source_crop=crop_rel_path,

                status="ok",

                raw_text=top.raw_text,
                raw_script=top.raw_script,
                raw_iast=top.raw_iast,
                transliteration_input=(
                    top.transliteration_input
                ),
                devanagari_text=(
                    top.devanagari_text
                ),
                normalization_actions=(
                    top.normalization_actions
                ),

                hypotheses=hypotheses,
                quality=quality,

                review_required=bool(
                    quality.warnings
                ),
                runtime_ms=float(
                    provider_result.runtime_ms
                ),
                device_used=(
                    provider_result.device_used
                ),

                error=None,
            )

        except Exception as exc:

            result = HTRLineResult(
                line_id=line_id,
                reading_order=reading_order,
                source_crop=crop_rel_path,

                status="error",

                raw_text="",
                raw_script=(
                    provider.output_script
                ),
                raw_iast=None,
                transliteration_input=None,
                devanagari_text=None,

                hypotheses=[],
                quality=_empty_quality(
                    "HTR_INFERENCE_ERROR"
                ),

                review_required=True,
                runtime_ms=0.0,
                device_used=None,

                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        results.append(
            result
        )

    _apply_page_length_diagnostics(
        results
    )

    # Persist per-line artifacts after page-relative diagnostics are applied.
    for result in results:
        store.write_json(
            f"L5/lines/"
            f"{result.line_id}.json",
            asdict(result),
        )

    successful = [
        result
        for result in results
        if result.status == "ok"
    ]

    errors = [
        result
        for result in results
        if result.status != "ok"
    ]

    blank = [
        result
        for result in successful
        if not result.raw_text.strip()
    ]

    review = [
        result
        for result in results
        if result.review_required
    ]

    valid_devanagari = [
        result
        for result in successful
        if (
            result.quality.transliteration_valid
        )
    ]

    latin_residual = [
        result
        for result in successful
        if (
            result.quality.latin_residual_ratio
            or 0.0
        )
        > 0.0
    ]

    repetitive_lines = [
        result
        for result in successful
        if (
            "POSSIBLE_REPETITIVE_GENERATION"
            in result.quality.warnings
            or
            "CONSECUTIVE_TOKEN_REPETITION"
            in result.quality.warnings
        )
    ]

    length_anomaly_lines = [
        result
        for result in successful
        if (
            "PAGE_RELATIVE_SEQUENCE_LENGTH_ANOMALY"
            in result.quality.warnings
            or
            "N_BEST_LENGTH_DIVERGENCE"
            in result.quality.warnings
        )
    ]

    metrics: Dict[str, Any] = {
        "algorithm_version": (
            LAYER5_VERSION
        ),
        "provider": (
            provider.provider_id
        ),
        "model_id": (
            provider.model_id
        ),
        "output_script": (
            provider.output_script
        ),
        "device": (
            provider.device
        ),

        "stage4_num_lines": (
            len(
                stage4_manifest[
                    "lines"
                ]
            )
        ),

        "attempted_lines": (
            len(results)
        ),
        "recognized_lines": (
            len(successful)
        ),
        "error_lines": (
            len(errors)
        ),
        "blank_lines": (
            len(blank)
        ),
        "review_required_lines": (
            len(review)
        ),

        "mean_hypothesis_entropy": (
            _safe_mean(
                [
                    result.quality.hypothesis_entropy
                    for result
                    in successful
                ]
            )
        ),

        "mean_decoder_stability": (
            _safe_mean(
                [
                    result.quality.decoder_stability
                    for result
                    in successful
                ]
            )
        ),

        "mean_script_purity": (
            _safe_mean(
                [
                    result.quality.script_purity
                    for result
                    in successful
                ]
            )
        ),

        "mean_latin_residual_ratio": (
            _safe_mean(
                [
                    result.quality.latin_residual_ratio
                    for result
                    in successful
                ]
            )
        ),

        "mean_token_repetition_ratio": (
            _safe_mean(
                [
                    result.quality.token_repetition_ratio
                    for result
                    in successful
                ]
            )
        ),

        "mean_hypothesis_length_spread": (
            _safe_mean(
                [
                    result.quality.hypothesis_length_spread
                    for result
                    in successful
                ]
            )
        ),

        "mean_sequence_length_anomaly_score": (
            _safe_mean(
                [
                    result.quality.sequence_length_anomaly_score
                    for result
                    in successful
                ]
            )
        ),

        "valid_devanagari_lines": (
            len(
                valid_devanagari
            )
        ),

        "lines_with_latin_residual": (
            len(
                latin_residual
            )
        ),

        "possible_repetitive_generation_lines": (
            len(
                repetitive_lines
            )
        ),

        "sequence_length_anomaly_lines": (
            len(
                length_anomaly_lines
            )
        ),

        # Intentionally not produced yet.
        "htr_readiness_H": None,
        "cer": None,
        "wer": None,
        "ground_truth_available": False,
        "recognition_accuracy_calibrated": False,
    }

    ordered_successful = sorted(
        successful,
        key=lambda result: (
            result.reading_order
        ),
    )

    raw_page_text = "\n".join(
        result.raw_text
        for result
        in ordered_successful
    )

    devanagari_page_text = "\n".join(
        result.devanagari_text
        or ""
        for result
        in ordered_successful
    )

    with open(
        store.path(
            "L5",
            "page_transcription_raw.txt",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            raw_page_text
        )

    with open(
        store.path(
            "L5",
            "page_transcription_devanagari.txt",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            devanagari_page_text
        )

    if provider.output_script == "iast":

        with open(
            store.path(
                "L5",
                "page_transcription_iast.txt",
            ),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                raw_page_text
            )

        transliteration_input_page = "\n".join(
            result.transliteration_input
            or ""
            for result
            in ordered_successful
        )

        with open(
            store.path(
                "L5",
                "page_transliteration_input.txt",
            ),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                transliteration_input_page
            )

    manifest: Dict[str, Any] = {
        "stage": "stage5_htr",
        "version": LAYER5_VERSION,
        "run_id": store.run_id,

        "provider": {
            "provider_id": (
                provider.provider_id
            ),
            "model_id": (
                provider.model_id
            ),
            "output_script": (
                provider.output_script
            ),
            "device_info": (
                provider.device_info
            ),
        },

        "generation": {
            "num_beams": (
                num_beams
            ),
            "n_best": (
                n_best
            ),
            "max_output_length": (
                max_output_length
            ),
        },

        "iast_normalization": {
            "global_unicode_form": "NFC",
            "global_nfd_applied": False,
            "policy": (
                "Audit-preserving selective normalization only; "
                "no global decomposition and no silent correction "
                "of raw model output."
            ),
            "canonical_equivalence_rules": {
                "U+0155": "U+0072 + U+0301",
                "U+0154": "U+0052 + U+0301",
            },
            "validated_transliteration_alias_rules": {
                "U+006C + U+0325": "U+1E37",
            },
            "post_transliteration_residual_rules": {
                "U+0310": "U+0901",
            },
            "raw_iast_preserved": True,
            "transliteration_input_persisted": True,
            "normalization_actions_persisted": True,
        },

        "upstream_stage4": {
            "version": (
                stage4_manifest.get(
                    "version"
                )
            ),
            "segmentation_confidence": (
                stage4_manifest
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "segmentation_confidence"
                )
            ),
            "num_lines": (
                stage4_manifest
                .get(
                    "metrics",
                    {},
                )
                .get(
                    "num_lines"
                )
            ),
        },

        "lines": [
            asdict(result)
            for result
            in results
        ],

        "metrics": (
            metrics
        ),

        "validation_note": (
            "Stage 5 provider-comparison implementation with "
            "audit-preserving selective IAST preparation and residual repair. "
            "No final H(p), CER/WER, or scholarly acceptance "
            "is produced."
        ),
    }

    store.write_json(
        "L5/htr_manifest.json",
        manifest,
    )

    store.write_json(
        "L5/page_transcription.json",
        {
            "run_id": (
                store.run_id
            ),
            "provider": (
                provider.provider_id
            ),
            "model_id": (
                provider.model_id
            ),
            "output_script": (
                provider.output_script
            ),
            "raw": (
                raw_page_text
            ),
            "devanagari": (
                devanagari_page_text
            ),
            "lines": [
                {
                    "line_id": (
                        result.line_id
                    ),
                    "reading_order": (
                        result.reading_order
                    ),
                    "raw_text": (
                        result.raw_text
                    ),
                    "raw_script": (
                        result.raw_script
                    ),
                    "raw_iast": (
                        result.raw_iast
                    ),
                    "transliteration_input": (
                        result.transliteration_input
                    ),
                    "devanagari_text": (
                        result.devanagari_text
                    ),
                    "normalization_actions": (
                        result.normalization_actions
                    ),
                    "review_required": (
                        result.review_required
                    ),
                    "quality": (
                        asdict(
                            result.quality
                        )
                    ),
                }
                for result
                in ordered_successful
            ],
        },
    )

    return Layer5Output(
        lines=results,
        metrics=metrics,
        manifest=manifest,
    )


def run_layer5_htr_baseline(
    store: ArtifactStore,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    num_beams: int = 4,
    n_best: int = 3,
    max_output_length: int = 192,
    max_lines: Optional[int] = None,
) -> Layer5Output:
    """
    Backward-compatible Provider-A entry point.
    """

    return run_layer5_htr(
        store,
        provider_name=(
            "trocr_iast_baseline"
        ),
        model_id=model_id,
        device=device,
        num_beams=num_beams,
        n_best=n_best,
        max_output_length=(
            max_output_length
        ),
        max_lines=max_lines,
    )
