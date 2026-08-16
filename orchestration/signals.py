from __future__ import annotations


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_C_signal(layer1_metrics: dict) -> float:
    """
    Enhancement adequacy C(p).

    High when contrast improves and foreground/background separation remains
    within a plausible range for manuscript text.
    """
    contrast_gain = float(layer1_metrics.get("contrast_gain", 0.0))
    foreground_ratio = float(layer1_metrics.get("binary_foreground_ratio", 0.0))

    if 0.05 <= foreground_ratio <= 0.35:
        fg_score = 1.0
    elif 0.02 <= foreground_ratio <= 0.50:
        fg_score = 0.7
    else:
        fg_score = 0.3

    gain_score = clamp01(contrast_gain / 20.0)
    return round(clamp01(0.6 * gain_score + 0.4 * fg_score), 4)


def compute_B_signal(layer1_metrics: dict) -> float:
    """
    Provisional bleed-through severity B(p).

    Lower is better. The current prototype uses a proxy until a dedicated
    recto-verso interference model is introduced.
    """
    proxy = float(layer1_metrics.get("bleedthrough_proxy", 0.5))
    return round(clamp01(proxy), 4)


def compute_G_signal(layer2_metrics: dict) -> float:
    """
    Geometry / structural reliability G(p).

    In the current prototype this is an upstream structural-risk proxy derived
    from damage and uncertainty. It is intentionally kept separate from the
    Stage 4 segmentation-readiness signal S(p).
    """
    damage_ratio = float(layer2_metrics.get("damage_ratio", 0.0))
    uncertainty_ratio = float(layer2_metrics.get("uncertainty_ratio", 0.0))

    g = 1.0 - (0.55 * damage_ratio + 0.45 * uncertainty_ratio)
    return round(clamp01(g), 4)


def compute_L_signal(layer3_metrics: dict) -> float:
    """
    Layout readiness L(p).

    High when text-bearing regions exist, text coverage is plausible and the
    page is not excessively fragmented.
    """
    num_regions = float(layer3_metrics.get("num_regions", 0.0))
    text_coverage = float(layer3_metrics.get("text_coverage_ratio", 0.0))
    fragmentation = float(layer3_metrics.get("region_fragmentation", 1.0))

    region_score = 1.0 if num_regions >= 1 else 0.0

    if 0.05 <= text_coverage <= 0.85:
        coverage_score = 1.0
    elif 0.02 <= text_coverage <= 0.95:
        coverage_score = 0.7
    else:
        coverage_score = 0.3

    fragmentation_score = 1.0 - clamp01(fragmentation)

    l = 0.4 * region_score + 0.4 * coverage_score + 0.2 * fragmentation_score
    return round(clamp01(l), 4)


def compute_S_signal(layer4_metrics: dict) -> float:
    """
    Segmentation readiness S(p).

    Combines transparent Stage 4 evidence:
      - foreground-ink preservation,
      - line-height consistency,
      - mean line confidence,
      - boundary certainty,
      - absence of suspected merged regions.

    This remains a rule-based research prototype signal. The weights should
    later be calibrated from annotated manuscript pages.
    """
    losslessness = float(layer4_metrics.get("losslessness_score", 0.0))
    consistency = float(layer4_metrics.get("line_height_consistency", 0.0))
    line_confidence = float(layer4_metrics.get("mean_line_confidence", 0.0))
    boundary_uncertainty = float(
        layer4_metrics.get("mean_boundary_uncertainty", 1.0)
    )
    merged_regions = int(layer4_metrics.get("suspected_merged_regions", 0))
    num_lines = int(layer4_metrics.get("num_lines", 0))

    boundary_score = 1.0 - clamp01(boundary_uncertainty)
    merge_score = (
        1.0
        if merged_regions == 0
        else max(0.0, 1.0 - merged_regions / max(1, num_lines))
    )

    s = (
        0.30 * clamp01(losslessness)
        + 0.25 * clamp01(consistency)
        + 0.20 * clamp01(line_confidence)
        + 0.15 * boundary_score
        + 0.10 * merge_score
    )
    return round(clamp01(s), 4)


def compute_R_signal(
    C: float,
    B: float,
    G: float,
    L: float,
    weights: dict | None = None,
) -> float:
    """
    Pre-segmentation routing readiness R(p).

    R(p) intentionally summarizes Stages 1-3 only. Stage 4 contributes S(p)
    separately so a downstream segmentation result can validate or challenge
    an upstream structural concern instead of being hidden inside one average.
    """
    if weights is None:
        weights = {"w1": 0.30, "w2": 0.20, "w3": 0.25, "w4": 0.25}

    R = (
        weights["w1"] * C
        + weights["w2"] * (1.0 - B)
        + weights["w3"] * G
        + weights["w4"] * L
    )
    return round(clamp01(R), 4)


def extract_H_signal(stage5_readiness: dict) -> float:
    """
    Extract the already-computed Stage 5 HTR readiness H(p).

    IMPORTANT:
    This function does NOT calculate H(p). Stage 5F owns the H(p) evidence
    fusion. The orchestration layer only validates and carries that signal
    forward so model/decoder confidence is never silently redefined as H(p).
    """
    page = stage5_readiness.get("page", {})
    value = page.get("htr_readiness_H_page")

    if value is None:
        raise ValueError(
            "Stage 5 readiness artifact does not contain page-level H(p)."
        )

    return round(clamp01(float(value)), 4)


def extract_T_signal(stage6_trust: dict) -> float:
    """
    Extract the already-computed Stage 6F semantic/transcription trust T(p).

    IMPORTANT:
    Stage 6F owns T(p). The orchestration layer never recomputes it and never
    interprets it as probability, CER, WER or transcription accuracy.
    """
    metrics = stage6_trust.get("metrics", {}) or {}
    value = metrics.get("page_T")

    if value is None:
        value = stage6_trust.get("page_T")

    if value is None:
        raise ValueError(
            "Stage 6 trust artifact does not contain page-level T(p)."
        )

    return round(clamp01(float(value)), 4)
