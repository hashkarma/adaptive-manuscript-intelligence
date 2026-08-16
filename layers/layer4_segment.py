from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.artifact_store import ArtifactStore
from core.io_utils import write_image


LAYER4_VERSION = "2.1.0-physical-line-consolidation"


@dataclass
class LineSegment:
    line_id: str
    reading_order: int

    # Page-space physical-line band used for analytical segmentation.
    x: int
    y: int
    w: int
    h: int

    ink_pixels: int
    line_confidence: float
    sirorekha_score: float
    boundary_uncertainty: float

    # Stage 5 continues to consume crop_rel_path.
    # This now points to an HTR-safe crop with synthetic context padding.
    crop_rel_path: str

    # Exact analytical crop preserved separately for audit/debugging.
    core_crop_rel_path: str

    # HTR-safe crop metadata.
    htr_crop_width: int
    htr_crop_height: int
    htr_padding_x: int
    htr_padding_y: int

    # Raw row-profile peaks consolidated into this physical line.
    source_profile_peaks: List[int]


@dataclass
class Layer4Output:
    segmentation_mask_u8: np.ndarray
    uncertainty_map_u8: np.ndarray
    overlay_bgr: np.ndarray
    lines: List[LineSegment]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]


def _ensure_gray_u8(
    image: np.ndarray,
    name: str,
) -> np.ndarray:
    if image is None:
        raise ValueError(f"{name} is required")

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    if image.ndim != 2:
        raise ValueError(
            f"{name} must be grayscale or BGR"
        )

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    return image


def _ensure_ink_is_white(
    binary_u8: np.ndarray,
) -> np.ndarray:
    binary = _ensure_gray_u8(
        binary_u8,
        "binary_u8",
    )

    _, binary = cv2.threshold(
        binary,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    if float(
        np.mean(binary > 0)
    ) > 0.5:
        binary = cv2.bitwise_not(
            binary
        )

    return binary


def _normalise_mask(
    mask_u8: Optional[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    if mask_u8 is None:
        return np.full(
            shape,
            255,
            dtype=np.uint8,
        )

    mask = _ensure_gray_u8(
        mask_u8,
        "text_region_mask_u8",
    )

    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (
                shape[1],
                shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        (mask > 0)
        .astype(np.uint8)
        * 255
    )


def _smooth_profile(
    profile: np.ndarray,
    window: int,
) -> np.ndarray:
    window = max(
        3,
        int(window) | 1,
    )

    kernel = (
        np.ones(
            window,
            dtype=np.float32,
        )
        / float(window)
    )

    return np.convolve(
        profile.astype(np.float32),
        kernel,
        mode="same",
    )


def _candidate_peaks(
    profile: np.ndarray,
    min_height: float,
) -> List[int]:
    if profile.size < 3:
        return []

    peaks: List[int] = []

    for i in range(
        1,
        len(profile) - 1,
    ):
        if (
            profile[i] >= min_height
            and profile[i] >= profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)

    return peaks


def _non_maximum_suppression(
    peaks: Sequence[int],
    profile: np.ndarray,
    min_distance: int,
) -> List[int]:
    selected: List[int] = []

    for peak in sorted(
        peaks,
        key=lambda p: float(profile[p]),
        reverse=True,
    ):
        if all(
            abs(peak - existing)
            >= min_distance
            for existing in selected
        ):
            selected.append(
                int(peak)
            )

    return sorted(selected)


def _adjacent_peak_valley_diagnostic(
    profile: np.ndarray,
    left_peak: int,
    right_peak: int,
) -> Dict[str, Any]:
    if right_peak <= left_peak + 1:
        valley_y = (
            left_peak
            + right_peak
        ) // 2
        valley_value = float(
            profile[valley_y]
        )
    else:
        local = profile[
            left_peak:right_peak + 1
        ]
        local_index = int(
            np.argmin(local)
        )
        valley_y = (
            left_peak
            + local_index
        )
        valley_value = float(
            local[local_index]
        )

    reference = max(
        1.0,
        min(
            float(profile[left_peak]),
            float(profile[right_peak]),
        ),
    )

    valley_ratio = float(
        np.clip(
            valley_value
            / reference,
            0.0,
            1.5,
        )
    )

    return {
        "left_peak": int(left_peak),
        "right_peak": int(right_peak),
        "gap": int(
            right_peak
            - left_peak
        ),
        "valley_y": int(valley_y),
        "valley_value": round(
            valley_value,
            4,
        ),
        "valley_to_smaller_peak_ratio": round(
            valley_ratio,
            4,
        ),
    }


def _group_peaks_into_physical_lines(
    profile: np.ndarray,
    peaks: Sequence[int],
    *,
    max_intra_line_peak_distance: int,
    shallow_valley_ratio: float,
) -> Tuple[
    List[List[int]],
    List[Dict[str, Any]],
]:
    """
    Consolidate multiple row-profile peaks that belong to one physical
    Devanagari manuscript line.

    Why this is needed:
      A single physical line may create multiple horizontal-projection peaks
      from śirorekhā, glyph bodies, upper/lower modifiers and Stage-1 binary
      fragmentation. A true boundary between two physical text lines should
      normally contain a substantially deeper projection valley.

    Adjacent peaks are merged only when BOTH conditions hold:
      1. their vertical distance is small enough to plausibly belong to one
         physical line; and
      2. the valley between them remains shallow relative to the smaller peak.

    This prevents simple distance-only merging from collapsing genuinely close
    neighbouring manuscript lines.
    """
    if not peaks:
        return [], []

    groups: List[List[int]] = [
        [int(peaks[0])]
    ]
    diagnostics: List[
        Dict[str, Any]
    ] = []

    for left_peak, right_peak in zip(
        peaks[:-1],
        peaks[1:],
    ):
        diag = (
            _adjacent_peak_valley_diagnostic(
                profile,
                int(left_peak),
                int(right_peak),
            )
        )

        merge = (
            diag["gap"]
            <= max_intra_line_peak_distance
            and
            diag[
                "valley_to_smaller_peak_ratio"
            ]
            >= shallow_valley_ratio
        )

        diag["decision"] = (
            "same_physical_line"
            if merge
            else "separate_physical_lines"
        )

        diagnostics.append(
            diag
        )

        if merge:
            groups[-1].append(
                int(right_peak)
            )
        else:
            groups.append(
                [int(right_peak)]
            )

    return (
        groups,
        diagnostics,
    )


def _representative_peak(
    profile: np.ndarray,
    group: Sequence[int],
) -> int:
    return int(
        max(
            group,
            key=lambda p: float(
                profile[p]
            ),
        )
    )


def _find_outer_boundary(
    profile: np.ndarray,
    peak: int,
    direction: int,
    low_threshold: float,
    consecutive_low_rows: int = 3,
) -> int:
    i = int(peak)
    low_count = 0

    while 0 <= i < len(profile):
        if profile[i] <= low_threshold:
            low_count += 1

            if (
                low_count
                >= consecutive_low_rows
            ):
                if direction < 0:
                    return min(
                        len(profile),
                        i
                        + consecutive_low_rows,
                    )

                return max(
                    0,
                    i
                    - consecutive_low_rows
                    + 1,
                )
        else:
            low_count = 0

        i += direction

    return (
        0
        if direction < 0
        else len(profile)
    )


def _bands_from_peak_groups(
    profile: np.ndarray,
    peak_groups: Sequence[
        Sequence[int]
    ],
    outer_low_ratio: float,
) -> List[Tuple[int, int]]:
    if not peak_groups:
        return []

    max_value = (
        float(np.max(profile))
        if profile.size
        else 0.0
    )

    low_threshold = max(
        1.0,
        outer_low_ratio
        * max_value,
    )

    cuts: List[int] = []

    for left_group, right_group in zip(
        peak_groups[:-1],
        peak_groups[1:],
    ):
        # Use the right-most peak of the upper physical line and the
        # left-most peak of the lower physical line. This prevents an
        # intra-line secondary peak from becoming a false line boundary.
        left_peak = int(
            left_group[-1]
        )
        right_peak = int(
            right_group[0]
        )

        if (
            right_peak
            <= left_peak + 1
        ):
            cut = (
                left_peak
                + right_peak
            ) // 2
        else:
            local = profile[
                left_peak:
                right_peak + 1
            ]

            cut = (
                left_peak
                + int(
                    np.argmin(local)
                )
            )

            cut = max(
                left_peak + 1,
                min(
                    right_peak - 1,
                    cut,
                ),
            )

        cuts.append(
            int(cut)
        )

    # Outer boundaries deliberately start from the extreme raw peak in the
    # first/last physical group, not merely from its strongest representative.
    start = _find_outer_boundary(
        profile,
        int(
            peak_groups[0][0]
        ),
        -1,
        low_threshold,
    )

    end = _find_outer_boundary(
        profile,
        int(
            peak_groups[-1][-1]
        ),
        +1,
        low_threshold,
    )

    boundaries = [
        int(start),
        *cuts,
        int(end),
    ]

    bands: List[
        Tuple[int, int]
    ] = []

    for i in range(
        len(boundaries) - 1
    ):
        y0 = int(
            max(
                0,
                boundaries[i],
            )
        )

        y1 = int(
            min(
                len(profile),
                boundaries[i + 1],
            )
        )

        if y1 > y0:
            bands.append(
                (
                    y0,
                    y1,
                )
            )

    return bands


def _estimate_sirorekha_score(
    line_ink: np.ndarray,
) -> float:
    h, w = line_ink.shape

    if h == 0 or w == 0:
        return 0.0

    upper_end = max(
        1,
        int(
            round(
                h * 0.55
            )
        ),
    )

    upper = (
        line_ink[:upper_end]
        > 0
    )

    occupancy = (
        np.sum(
            upper,
            axis=1,
        ).astype(np.float32)
        / float(
            max(
                1,
                w,
            )
        )
    )

    strongest = (
        float(
            np.max(occupancy)
        )
        if occupancy.size
        else 0.0
    )

    return float(
        np.clip(
            strongest / 0.18,
            0.0,
            1.0,
        )
    )


def _boundary_uncertainty(
    profile: np.ndarray,
    y_start: int,
    y_end: int,
    peak_value: float,
    radius: int = 2,
) -> float:
    if peak_value <= 0:
        return 1.0

    indices: List[int] = []

    for centre in (
        y_start,
        max(
            y_start,
            y_end - 1,
        ),
    ):
        indices.extend(
            range(
                max(
                    0,
                    centre - radius,
                ),
                min(
                    len(profile),
                    centre
                    + radius
                    + 1,
                ),
            )
        )

    if not indices:
        return 1.0

    density = float(
        np.mean(
            profile[indices]
        )
    )

    return float(
        np.clip(
            density / peak_value,
            0.0,
            1.0,
        )
    )


def _line_consistency_score(
    heights: List[int],
) -> float:
    if not heights:
        return 0.0

    if len(heights) == 1:
        return 0.75

    median_height = float(
        np.median(heights)
    )

    if median_height <= 0:
        return 0.0

    median_absolute_deviation = float(
        np.median(
            np.abs(
                np.asarray(heights)
                - median_height
            )
        )
    )

    robust_cv = (
        median_absolute_deviation
        / median_height
    )

    return float(
        np.clip(
            1.0
            - 2.0
            * robust_cv,
            0.0,
            1.0,
        )
    )


def _make_htr_safe_crop(
    line_gray: np.ndarray,
    *,
    pad_x: int,
    pad_y: int,
    background_value: int,
) -> np.ndarray:
    """
    Add synthetic neutral context instead of stealing pixels from a neighbouring
    line. This is safer than expanding a crop across an inter-line valley.
    """
    return cv2.copyMakeBorder(
        line_gray,
        int(pad_y),
        int(pad_y),
        int(pad_x),
        int(pad_x),
        borderType=cv2.BORDER_CONSTANT,
        value=int(background_value),
    )


def _render_profile_debug(
    profile: np.ndarray,
    raw_peaks: Sequence[int],
    peak_groups: Sequence[
        Sequence[int]
    ],
    bands: Sequence[
        Tuple[int, int]
    ],
    width: int = 900,
) -> np.ndarray:
    height = max(
        300,
        len(profile),
    )

    canvas = np.full(
        (
            height,
            width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    max_value = (
        float(np.max(profile))
        if profile.size
        else 1.0
    )

    max_value = max(
        max_value,
        1.0,
    )

    points = []

    for y, value in enumerate(
        profile
    ):
        x = (
            int(
                round(
                    (
                        float(value)
                        / max_value
                    )
                    * (
                        width - 80
                    )
                )
            )
            + 40
        )

        points.append(
            (
                x,
                y,
            )
        )

    if len(points) >= 2:
        cv2.polylines(
            canvas,
            [
                np.asarray(
                    points,
                    dtype=np.int32,
                )
            ],
            False,
            (
                80,
                80,
                80,
            ),
            1,
        )

    # Raw peaks are short orange ticks. They are evidence, not physical lines.
    for peak in raw_peaks:
        cv2.line(
            canvas,
            (
                0,
                int(peak),
            ),
            (
                28,
                int(peak),
            ),
            (
                0,
                140,
                255,
            ),
            1,
        )

        cv2.putText(
            canvas,
            f"r{peak}",
            (
                30,
                max(
                    12,
                    int(peak) - 2,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (
                0,
                110,
                210,
            ),
            1,
        )

    representatives = [
        _representative_peak(
            profile,
            group,
        )
        for group in peak_groups
    ]

    # Consolidated physical-line representatives remain full red lines.
    for index, (
        representative,
        group,
    ) in enumerate(
        zip(
            representatives,
            peak_groups,
        ),
        start=1,
    ):
        cv2.line(
            canvas,
            (
                0,
                int(representative),
            ),
            (
                width - 1,
                int(representative),
            ),
            (
                0,
                0,
                255,
            ),
            1,
        )

        group_text = ",".join(
            str(int(p))
            for p in group
        )

        cv2.putText(
            canvas,
            f"G{index}[{group_text}]",
            (
                5,
                max(
                    12,
                    int(representative)
                    - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (
                0,
                0,
                180,
            ),
            1,
        )

    for index, (
        y0,
        y1,
    ) in enumerate(
        bands,
        start=1,
    ):
        cv2.line(
            canvas,
            (
                0,
                int(y0),
            ),
            (
                width - 1,
                int(y0),
            ),
            (
                0,
                180,
                0,
            ),
            1,
        )

        cv2.line(
            canvas,
            (
                0,
                int(y1),
            ),
            (
                width - 1,
                int(y1),
            ),
            (
                0,
                180,
                0,
            ),
            1,
        )

        cv2.putText(
            canvas,
            str(index),
            (
                width - 35,
                int(
                    (
                        y0 + y1
                    )
                    / 2
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                120,
                0,
            ),
            1,
        )

    return canvas


def run_layer4_segmentation(
    balanced_gray_u8: np.ndarray,
    binary_u8: np.ndarray,
    store: ArtifactStore,
    *,
    text_region_mask_u8: Optional[
        np.ndarray
    ] = None,
    upstream_uncertainty_u8: Optional[
        np.ndarray
    ] = None,
    profile_smooth_window: int = 5,
    peak_threshold_ratio: float = 0.25,
    peak_min_distance_ratio: float = 0.038,
    outer_low_ratio: float = 0.04,
    minimum_line_height_ratio: float = 0.02,
    crop_padding_x: int = 8,
    crop_padding_y: int = 4,
    physical_peak_merge_distance_ratio: float = 0.08,
    shallow_valley_ratio: float = 0.45,
    htr_safe_padding_y_ratio: float = 0.12,
    htr_safe_padding_x: int = 12,
) -> Layer4Output:
    """
    Research Stage 4:
    loss-audited physical-line segmentation with shallow-valley peak
    consolidation and HTR-safe contextual crops.

    Important:
    raw projection peaks are treated as geometric evidence, not one-to-one
    text-line declarations.
    """
    gray = _ensure_gray_u8(
        balanced_gray_u8,
        "balanced_gray_u8",
    )

    ink = _ensure_ink_is_white(
        binary_u8
    )

    if gray.shape != ink.shape:
        raise ValueError(
            "Image shape mismatch: "
            f"gray={gray.shape}, "
            f"binary={ink.shape}"
        )

    h, w = gray.shape

    region_mask = _normalise_mask(
        text_region_mask_u8,
        gray.shape,
    )

    considered_ink = cv2.bitwise_and(
        ink,
        region_mask,
    )

    total_page_ink = int(
        np.count_nonzero(ink)
    )

    total_considered_ink = int(
        np.count_nonzero(
            considered_ink
        )
    )

    outside_layout_ink = max(
        0,
        total_page_ink
        - total_considered_ink,
    )

    outside_layout_ratio = (
        outside_layout_ink
        / float(total_page_ink)
        if total_page_ink
        else 0.0
    )

    raw_profile = np.sum(
        considered_ink > 0,
        axis=1,
    ).astype(np.float32)

    smooth_profile = _smooth_profile(
        raw_profile,
        profile_smooth_window,
    )

    max_profile = (
        float(
            np.max(
                smooth_profile
            )
        )
        if smooth_profile.size
        else 0.0
    )

    minimum_peak_height = (
        peak_threshold_ratio
        * max_profile
    )

    minimum_peak_distance = max(
        8,
        int(
            round(
                h
                * peak_min_distance_ratio
            )
        ),
    )

    candidates = _candidate_peaks(
        smooth_profile,
        minimum_peak_height,
    )

    raw_peaks = (
        _non_maximum_suppression(
            candidates,
            smooth_profile,
            minimum_peak_distance,
        )
    )

    max_intra_line_peak_distance = max(
        12,
        int(
            round(
                h
                * physical_peak_merge_distance_ratio
            )
        ),
    )

    peak_groups, pair_diagnostics = (
        _group_peaks_into_physical_lines(
            smooth_profile,
            raw_peaks,
            max_intra_line_peak_distance=(
                max_intra_line_peak_distance
            ),
            shallow_valley_ratio=(
                shallow_valley_ratio
            ),
        )
    )

    physical_bands = (
        _bands_from_peak_groups(
            smooth_profile,
            peak_groups,
            outer_low_ratio,
        )
    )

    minimum_line_height = max(
        3,
        int(
            round(
                h
                * minimum_line_height_ratio
            )
        ),
    )

    filtered: List[
        Tuple[
            List[int],
            int,
            int,
            int,
        ]
    ] = []

    for group, (
        y0,
        y1,
    ) in zip(
        peak_groups,
        physical_bands,
    ):
        if (
            y1 - y0
            < minimum_line_height
        ):
            continue

        if int(
            np.count_nonzero(
                considered_ink[
                    y0:y1
                ]
            )
        ) == 0:
            continue

        representative = (
            _representative_peak(
                smooth_profile,
                group,
            )
        )

        filtered.append(
            (
                list(group),
                representative,
                int(y0),
                int(y1),
            )
        )

    segmentation_mask = (
        np.zeros_like(ink)
    )

    assignment_count = (
        np.zeros_like(
            ink,
            dtype=np.uint16,
        )
    )

    uncertainty_map = (
        np.zeros_like(ink)
    )

    overlay = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    lines: List[
        LineSegment
    ] = []

    background_value = int(
        np.clip(
            np.percentile(
                gray,
                95,
            ),
            220,
            255,
        )
    )

    for (
        group,
        representative_peak,
        band_start,
        band_end,
    ) in filtered:
        band_ink_full = (
            considered_ink[
                band_start:
                band_end
            ]
        )

        _, xs = np.where(
            band_ink_full > 0
        )

        if xs.size == 0:
            continue

        x_start = max(
            0,
            int(
                xs.min()
            )
            - crop_padding_x,
        )

        x_end = min(
            w,
            int(
                xs.max()
            )
            + 1
            + crop_padding_x,
        )

        core_gray = gray[
            band_start:
            band_end,
            x_start:
            x_end,
        ]

        core_ink = (
            considered_ink[
                band_start:
                band_end,
                x_start:
                x_end,
            ]
        )

        line_id = (
            f"line_"
            f"{len(lines) + 1:03d}"
        )

        core_crop_rel_path = (
            f"L4/core_lines/"
            f"{line_id}.png"
        )

        crop_rel_path = (
            f"L4/lines/"
            f"{line_id}.png"
        )

        write_image(
            store.path(
                core_crop_rel_path
            ),
            core_gray,
        )

        adaptive_htr_pad_y = max(
            int(crop_padding_y),
            int(
                round(
                    (
                        band_end
                        - band_start
                    )
                    * htr_safe_padding_y_ratio
                )
            ),
        )

        htr_crop = (
            _make_htr_safe_crop(
                core_gray,
                pad_x=max(
                    0,
                    int(
                        htr_safe_padding_x
                    ),
                ),
                pad_y=max(
                    0,
                    adaptive_htr_pad_y,
                ),
                background_value=(
                    background_value
                ),
            )
        )

        write_image(
            store.path(
                crop_rel_path
            ),
            htr_crop,
        )

        core = (
            considered_ink[
                band_start:
                band_end,
                x_start:
                x_end,
            ]
            > 0
        )

        assignment_count[
            band_start:
            band_end,
            x_start:
            x_end,
        ][core] += 1

        segmentation_mask[
            band_start:
            band_end,
            x_start:
            x_end,
        ][core] = 255

        peak_value = max(
            float(
                smooth_profile[
                    representative_peak
                ]
            ),
            1.0,
        )

        boundary_score = (
            _boundary_uncertainty(
                smooth_profile,
                band_start,
                band_end,
                peak_value,
            )
        )

        sirorekha_score = (
            _estimate_sirorekha_score(
                core_ink
            )
        )

        density_score = float(
            np.clip(
                (
                    np.count_nonzero(
                        core_ink
                    )
                    / float(
                        max(
                            1,
                            core_ink.size,
                        )
                    )
                )
                / 0.22,
                0.0,
                1.0,
            )
        )

        line_confidence = float(
            np.clip(
                0.45
                * (
                    1.0
                    - boundary_score
                )
                + 0.30
                * sirorekha_score
                + 0.25
                * density_score,
                0.0,
                1.0,
            )
        )

        cv2.rectangle(
            overlay,
            (
                x_start,
                band_start,
            ),
            (
                x_end - 1,
                band_end - 1,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        cv2.putText(
            overlay,
            str(
                len(lines) + 1
            ),
            (
                x_start + 4,
                max(
                    14,
                    band_start + 16,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                0,
                0,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        radius = 2

        for boundary in (
            band_start,
            band_end,
        ):
            yb0 = max(
                0,
                boundary - radius,
            )

            yb1 = min(
                h,
                boundary
                + radius
                + 1,
            )

            uncertainty_map[
                yb0:yb1,
                x_start:x_end,
            ] = np.maximum(
                uncertainty_map[
                    yb0:yb1,
                    x_start:x_end,
                ],
                int(
                    round(
                        255
                        * boundary_score
                    )
                ),
            )

        lines.append(
            LineSegment(
                line_id=line_id,
                reading_order=(
                    len(lines) + 1
                ),
                x=x_start,
                y=band_start,
                w=(
                    x_end
                    - x_start
                ),
                h=(
                    band_end
                    - band_start
                ),
                ink_pixels=int(
                    np.count_nonzero(
                        core_ink
                    )
                ),
                line_confidence=round(
                    line_confidence,
                    4,
                ),
                sirorekha_score=round(
                    sirorekha_score,
                    4,
                ),
                boundary_uncertainty=round(
                    boundary_score,
                    4,
                ),
                crop_rel_path=(
                    crop_rel_path
                ),
                core_crop_rel_path=(
                    core_crop_rel_path
                ),
                htr_crop_width=int(
                    htr_crop.shape[1]
                ),
                htr_crop_height=int(
                    htr_crop.shape[0]
                ),
                htr_padding_x=int(
                    max(
                        0,
                        htr_safe_padding_x,
                    )
                ),
                htr_padding_y=int(
                    max(
                        0,
                        adaptive_htr_pad_y,
                    )
                ),
                source_profile_peaks=[
                    int(p)
                    for p in group
                ],
            )
        )

    orphan_mask = (
        (considered_ink > 0)
        & (
            assignment_count
            == 0
        )
    )

    duplicate_mask = (
        assignment_count > 1
    )

    uncertainty_map[
        orphan_mask
    ] = 255

    if (
        upstream_uncertainty_u8
        is not None
    ):
        upstream = (
            _ensure_gray_u8(
                upstream_uncertainty_u8,
                "upstream_uncertainty_u8",
            )
        )

        if (
            upstream.shape
            != gray.shape
        ):
            upstream = cv2.resize(
                upstream,
                (
                    w,
                    h,
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

        uncertainty_map = (
            np.maximum(
                uncertainty_map,
                upstream,
            )
        )

    assigned_ink = int(
        np.count_nonzero(
            (
                considered_ink
                > 0
            )
            & (
                assignment_count
                > 0
            )
        )
    )

    orphan_ink = int(
        np.count_nonzero(
            orphan_mask
        )
    )

    duplicate_ink = int(
        np.count_nonzero(
            duplicate_mask
        )
    )

    losslessness_score = (
        assigned_ink
        / float(
            total_considered_ink
        )
        if total_considered_ink
        else 0.0
    )

    orphan_ink_ratio = (
        orphan_ink
        / float(
            total_considered_ink
        )
        if total_considered_ink
        else 0.0
    )

    duplicate_assignment_ratio = (
        duplicate_ink
        / float(
            total_considered_ink
        )
        if total_considered_ink
        else 0.0
    )

    # Use analytical physical-band heights. Synthetic HTR padding must not
    # distort segmentation-quality metrics.
    heights = [
        int(line.h)
        for line in lines
    ]

    median_height = (
        float(
            np.median(
                heights
            )
        )
        if heights
        else 0.0
    )

    max_height_ratio = (
        max(heights)
        / median_height
        if median_height > 0
        else 0.0
    )

    consistency = (
        _line_consistency_score(
            heights
        )
    )

    suspected_merged_regions = int(
        sum(
            1
            for height in heights
            if (
                median_height > 0
                and height
                > 1.55
                * median_height
            )
        )
    )

    mean_line_confidence = (
        float(
            np.mean(
                [
                    line.line_confidence
                    for line in lines
                ]
            )
        )
        if lines
        else 0.0
    )

    mean_sirorekha = (
        float(
            np.mean(
                [
                    line.sirorekha_score
                    for line in lines
                ]
            )
        )
        if lines
        else 0.0
    )

    mean_boundary_uncertainty = (
        float(
            np.mean(
                [
                    line.boundary_uncertainty
                    for line in lines
                ]
            )
        )
        if lines
        else 1.0
    )

    merge_penalty = min(
        1.0,
        suspected_merged_regions
        / float(
            max(
                1,
                len(lines),
            )
        ),
    )

    segmentation_confidence = float(
        np.clip(
            0.30
            * losslessness_score
            + 0.25
            * consistency
            + 0.20
            * mean_line_confidence
            + 0.15
            * (
                1.0
                - mean_boundary_uncertainty
            )
            + 0.10
            * (
                1.0
                - merge_penalty
            ),
            0.0,
            1.0,
        )
    )

    write_image(
        store.path(
            "L4",
            "segmentation_mask.png",
        ),
        segmentation_mask,
    )

    write_image(
        store.path(
            "L4",
            "segmentation_overlay.png",
        ),
        overlay,
    )

    write_image(
        store.path(
            "L4",
            "uncertainty_map.png",
        ),
        uncertainty_map,
    )

    write_image(
        store.path(
            "L4",
            "orphan_ink_mask.png",
        ),
        (
            orphan_mask
            .astype(np.uint8)
            * 255
        ),
    )

    filtered_groups = [
        item[0]
        for item in filtered
    ]

    filtered_bands = [
        (
            item[2],
            item[3],
        )
        for item in filtered
    ]

    debug_image = (
        _render_profile_debug(
            smooth_profile,
            raw_peaks,
            filtered_groups,
            filtered_bands,
        )
    )

    write_image(
        store.path(
            "L4",
            "row_profile_debug.png",
        ),
        debug_image,
    )

    multi_peak_groups = int(
        sum(
            1
            for group in filtered_groups
            if len(group) > 1
        )
    )

    intra_line_peak_merges = int(
        sum(
            max(
                0,
                len(group) - 1,
            )
            for group in filtered_groups
        )
    )

    metrics: Dict[str, Any] = {
        "algorithm_version": (
            LAYER4_VERSION
        ),
        "algorithm": (
            "row-profile peak detection with shallow-valley physical-line "
            "consolidation, inter-line valley boundaries and HTR-safe crops"
        ),
        "num_lines": len(lines),

        # Backward compatibility: this remains the number of raw NMS peaks.
        "num_profile_peaks": len(
            raw_peaks
        ),
        "num_profile_peaks_raw": len(
            raw_peaks
        ),
        "num_physical_line_groups": len(
            filtered_groups
        ),
        "num_intra_line_peak_merges": (
            intra_line_peak_merges
        ),
        "multi_peak_physical_line_groups": (
            multi_peak_groups
        ),
        "peak_consolidation_applied": (
            intra_line_peak_merges > 0
        ),

        "losslessness_score": round(
            losslessness_score,
            4,
        ),
        "orphan_ink_ratio": round(
            orphan_ink_ratio,
            4,
        ),
        "duplicate_assignment_ratio": round(
            duplicate_assignment_ratio,
            4,
        ),
        "outside_layout_ink_ratio": round(
            outside_layout_ratio,
            4,
        ),
        "mean_line_confidence": round(
            mean_line_confidence,
            4,
        ),
        "mean_sirorekha_evidence": round(
            mean_sirorekha,
            4,
        ),
        "mean_boundary_uncertainty": round(
            mean_boundary_uncertainty,
            4,
        ),
        "line_height_consistency": round(
            consistency,
            4,
        ),
        "median_line_height": round(
            median_height,
            2,
        ),
        "max_height_to_median_ratio": round(
            max_height_ratio,
            4,
        ),
        "suspected_merged_regions": (
            suspected_merged_regions
        ),
        "segmentation_confidence": round(
            segmentation_confidence,
            4,
        ),
        "total_ink_pixels_considered": (
            total_considered_ink
        ),
        "assigned_ink_pixels": (
            assigned_ink
        ),
        "orphan_ink_pixels": (
            orphan_ink
        ),
    }

    manifest = {
        "stage": (
            "stage4_segmentation"
        ),
        "version": (
            LAYER4_VERSION
        ),
        "run_id": (
            store.run_id
        ),
        "reading_order": (
            "top_to_bottom"
        ),
        "lines": [
            asdict(line)
            for line in lines
        ],
        "metrics": (
            metrics
        ),
        "validation_note": (
            "Raw row-profile peaks are geometric evidence and may represent "
            "śirorekhā/body/modifier structure within one physical line. "
            "Stage 4 therefore consolidates shallow-valley peak groups before "
            "declaring physical manuscript lines. HTR-safe crops add synthetic "
            "context padding without importing pixels from neighbouring lines."
        ),
    }

    store.write_json(
        "L4/line_manifest.json",
        manifest,
    )

    debug: Dict[
        str,
        Any,
    ] = {
        "version": (
            LAYER4_VERSION
        ),
        "parameters": {
            "profile_smooth_window": (
                profile_smooth_window
            ),
            "peak_threshold_ratio": (
                peak_threshold_ratio
            ),
            "peak_min_distance_ratio": (
                peak_min_distance_ratio
            ),
            "outer_low_ratio": (
                outer_low_ratio
            ),
            "minimum_line_height_ratio": (
                minimum_line_height_ratio
            ),
            "crop_padding_x": (
                crop_padding_x
            ),
            "crop_padding_y": (
                crop_padding_y
            ),
            "physical_peak_merge_distance_ratio": (
                physical_peak_merge_distance_ratio
            ),
            "max_intra_line_peak_distance": (
                max_intra_line_peak_distance
            ),
            "shallow_valley_ratio": (
                shallow_valley_ratio
            ),
            "htr_safe_padding_y_ratio": (
                htr_safe_padding_y_ratio
            ),
            "htr_safe_padding_x": (
                htr_safe_padding_x
            ),
        },
        "raw_peaks": [
            int(p)
            for p in raw_peaks
        ],
        "peak_groups": [
            [
                int(p)
                for p in group
            ]
            for group in filtered_groups
        ],
        "representative_peaks": [
            _representative_peak(
                smooth_profile,
                group,
            )
            for group in filtered_groups
        ],
        "bands": [
            [
                int(y0),
                int(y1),
            ]
            for (
                y0,
                y1,
            ) in filtered_bands
        ],
        "adjacent_peak_diagnostics": (
            pair_diagnostics
        ),
    }

    store.write_json(
        "L4/segmentation_debug.json",
        debug,
    )

    return Layer4Output(
        segmentation_mask_u8=(
            segmentation_mask
        ),
        uncertainty_map_u8=(
            uncertainty_map
        ),
        overlay_bgr=(
            overlay
        ),
        lines=(
            lines
        ),
        metrics=(
            metrics
        ),
        debug=(
            debug
        ),
    )
