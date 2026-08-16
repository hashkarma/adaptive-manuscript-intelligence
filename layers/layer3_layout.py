from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from core.io_utils import write_image
from core.artifact_store import ArtifactStore


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class Layer3Output:
    text_region_mask_u8: np.ndarray
    regions: List[TextRegion]
    overlay_bgr: np.ndarray
    metrics: Dict[str, Any]
    debug: Dict[str, Any]


def _ensure_ink_is_white(binary_u8: np.ndarray) -> np.ndarray:
    white_ratio = float(np.mean(binary_u8 > 127))
    if white_ratio > 0.5:
        return cv2.bitwise_not(binary_u8)
    return binary_u8


def _smooth_boolean_1d(arr: np.ndarray, k: int = 9) -> np.ndarray:
    k = max(3, int(k) | 1)
    x = arr.astype(np.uint8)
    kernel = np.ones((k,), dtype=np.float32) / k
    smoothed = np.convolve(x, kernel, mode="same")
    return smoothed > 0.35


def _merge_close_bands(bands: List[Tuple[int, int]], gap: int = 10) -> List[Tuple[int, int]]:
    if not bands:
        return []
    merged = [bands[0]]
    for (s, e) in bands[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def run_layer3_layout(
    balanced_gray_u8: np.ndarray,
    binary_u8: np.ndarray,
    store: ArtifactStore,
    *,
    row_density_threshold_ratio: float = 0.18,
    min_band_height_ratio: float = 0.02,
    smooth_window: int = 11,
    merge_gap_pixels: int = 12,
) -> Layer3Output:
    H, W = binary_u8.shape
    ink_white = _ensure_ink_is_white(binary_u8)

    ink_per_row = np.sum(ink_white > 0, axis=1).astype(np.float32)
    max_ink = float(np.max(ink_per_row))

    if max_ink <= 1e-6:
        region_mask = np.zeros((H, W), dtype=np.uint8)
        overlay = cv2.cvtColor(balanced_gray_u8, cv2.COLOR_GRAY2BGR)

        write_image(store.path("L3", "text_region_mask.png"), region_mask)
        write_image(store.path("L3", "layout_overlay.png"), overlay)

        metrics = {
            "num_regions": 0,
            "text_coverage_ratio": 0.0,
            "mean_band_height": 0.0,
            "region_fragmentation": 1.0,
        }

        return Layer3Output(
            text_region_mask_u8=region_mask,
            regions=[],
            overlay_bgr=overlay,
            metrics=metrics,
            debug={"notes": "No ink detected in binary image."}
        )

    threshold = row_density_threshold_ratio * max_ink
    text_rows = ink_per_row > threshold
    text_rows_smooth = _smooth_boolean_1d(text_rows, k=smooth_window)

    bands: List[Tuple[int, int]] = []
    y = 0
    while y < H:
        if text_rows_smooth[y]:
            y_start = y
            while y < H and text_rows_smooth[y]:
                y += 1
            y_end = y
            bands.append((y_start, y_end))
        else:
            y += 1

    bands = _merge_close_bands(bands, gap=merge_gap_pixels)

    min_band_height = max(2, int(min_band_height_ratio * H))
    filtered_bands = [(s, e) for (s, e) in bands if (e - s) >= min_band_height]

    regions: List[TextRegion] = []
    region_mask = np.zeros((H, W), dtype=np.uint8)

    for (s, e) in filtered_bands:
        h = e - s
        regions.append(TextRegion(x=0, y=s, w=W, h=h, area=W * h))
        region_mask[s:e, 0:W] = 255

    overlay = cv2.cvtColor(balanced_gray_u8, cv2.COLOR_GRAY2BGR)
    for r in regions:
        cv2.rectangle(
            overlay,
            (r.x, r.y),
            (r.x + r.w - 1, r.y + r.h - 1),
            (0, 255, 0),
            2
        )

    text_coverage_ratio = float(np.mean(region_mask > 0))
    region_fragmentation = 1.0 / (1.0 + len(regions)) if len(regions) > 0 else 1.0

    write_image(store.path("L3", "text_region_mask.png"), region_mask)
    write_image(store.path("L3", "layout_overlay.png"), overlay)

    metrics = {
        "num_regions": len(regions),
        "text_coverage_ratio": round(text_coverage_ratio, 4),
        "mean_band_height": round(float(np.mean([r.h for r in regions])) if regions else 0.0, 3),
        "region_fragmentation": round(region_fragmentation, 4),
    }

    debug = {
        "notes": (
            "Layer 3 detects text-bearing structure and contributes primarily to layout readiness L(p)."
        )
    }

    return Layer3Output(
        text_region_mask_u8=region_mask,
        regions=regions,
        overlay_bgr=overlay,
        metrics=metrics,
        debug=debug
    )