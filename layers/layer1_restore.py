from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np

from core.io_utils import bgr_to_gray, safe_uint8, write_image
from core.artifact_store import ArtifactStore


@dataclass
class Layer1Output:
    tone_bgr: np.ndarray
    balanced_bgr: np.ndarray
    binary_u8: np.ndarray
    separator_mask_u8: np.ndarray
    metrics: Dict[str, Any]
    debug: Dict[str, Any]


def _estimate_background(gray_u8: np.ndarray, ksize: int = 35) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    return cv2.GaussianBlur(gray_u8, (k, k), 0)


def _flatten_illumination(gray_u8: np.ndarray, bg_u8: np.ndarray) -> np.ndarray:
    eps = 1e-6
    g = gray_u8.astype(np.float32)
    b = bg_u8.astype(np.float32) + eps
    norm = (g / b) * 180.0
    return safe_uint8(norm)


def _ink_balance(gray_u8: np.ndarray, clip_limit: float = 2.0, tile: int = 16, gamma: float = 1.0) -> np.ndarray:
    tile = max(4, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile, tile))
    c = clahe.apply(gray_u8)

    gamma = max(0.2, float(gamma))
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(c, lut)


def _detect_orange_red_separators(bgr: np.ndarray, sat_min: int = 60, val_min: int = 40) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, sat_min, val_min])
    upper_red1 = np.array([12, 255, 255])

    lower_red2 = np.array([160, sat_min, val_min])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _binarize_for_structure(gray_u8: np.ndarray, method: str = "otsu", block: int = 35, C: int = 12) -> np.ndarray:
    if method == "adaptive":
        block = max(3, int(block) | 1)
        return cv2.adaptiveThreshold(
            gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, int(C)
        )

    _, bin_img = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img


def run_layer1_restore(
    raw_bgr: np.ndarray,
    store: ArtifactStore,
    *,
    bg_ksize: int = 35,
    clahe_clip: float = 2.0,
    clahe_tile: int = 16,
    gamma: float = 1.0,
    bin_method: str = "otsu",
    adaptive_block: int = 35,
    adaptive_C: int = 12,
    preserve_separators: bool = True,
) -> Layer1Output:
    gray = bgr_to_gray(raw_bgr)
    tone = raw_bgr.copy()

    contrast_before = float(np.std(gray))
    brightness_before = float(np.mean(gray))

    bg = _estimate_background(gray, ksize=bg_ksize)
    flat = _flatten_illumination(gray, bg)
    balanced_gray = _ink_balance(flat, clip_limit=clahe_clip, tile=clahe_tile, gamma=gamma)
    balanced_bgr = cv2.cvtColor(balanced_gray, cv2.COLOR_GRAY2BGR)

    contrast_after = float(np.std(balanced_gray))
    brightness_after = float(np.mean(balanced_gray))
    contrast_gain = contrast_after - contrast_before

    sep_mask = _detect_orange_red_separators(raw_bgr) if preserve_separators else np.zeros_like(gray)

    binary = _binarize_for_structure(
        balanced_gray, method=bin_method, block=adaptive_block, C=adaptive_C
    )

    # keep black text on white
    if np.mean(binary) < 127:
        binary = 255 - binary

    separator_pixels_detected = int(np.sum(sep_mask > 0))
    binary_foreground_ratio = float(np.mean(binary < 128))

    # provisional bleed-through estimate:
    # darker/noisier background + unusually dense binary often indicates possible interference
    background_residual = cv2.absdiff(gray, bg)
    bleedthrough_proxy = float(np.mean(background_residual)) / 255.0
    bleedthrough_proxy = min(max(bleedthrough_proxy, 0.0), 1.0)

    write_image(store.path("L1", "tone.png"), tone)
    write_image(store.path("L1", "balanced.png"), balanced_bgr)
    write_image(store.path("L1", "binary.png"), binary)
    write_image(store.path("L1", "separator_mask.png"), sep_mask)
    write_image(store.path("L1", "debug_background.png"), bg)
    write_image(store.path("L1", "debug_flat.png"), flat)
    write_image(store.path("L1", "debug_balanced_gray.png"), balanced_gray)

    metrics = {
        "contrast_before": round(contrast_before, 3),
        "contrast_after": round(contrast_after, 3),
        "contrast_gain": round(contrast_gain, 3),
        "brightness_before": round(brightness_before, 3),
        "brightness_after": round(brightness_after, 3),
        "binary_foreground_ratio": round(binary_foreground_ratio, 4),
        "separator_pixels_detected": separator_pixels_detected,
        "bleedthrough_proxy": round(bleedthrough_proxy, 4),
    }

    debug = {
        "notes": (
            "Layer 1 restores readability and emits metrics for enhancement adequacy C(p) "
            "and provisional bleed-through severity B(p)."
        )
    }

    return Layer1Output(
        tone_bgr=tone,
        balanced_bgr=balanced_bgr,
        binary_u8=binary,
        separator_mask_u8=sep_mask,
        metrics=metrics,
        debug=debug
    )