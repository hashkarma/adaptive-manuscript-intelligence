from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np

from core.io_utils import safe_uint8, write_image
from core.artifact_store import ArtifactStore


@dataclass
class Layer2Output:
    damage_mask_u8: np.ndarray
    uncertainty_u8: np.ndarray
    metrics: Dict[str, Any]
    debug: Dict[str, Any]


def _local_contrast_map(gray_u8: np.ndarray, win: int = 31) -> np.ndarray:
    win = max(3, int(win) | 1)
    mean = cv2.GaussianBlur(gray_u8, (win, win), 0)
    diff = cv2.absdiff(gray_u8, mean)
    contrast = cv2.GaussianBlur(diff, (win, win), 0)
    return contrast


def _detect_holes_and_tears(gray_u8: np.ndarray) -> np.ndarray:
    _, bright = cv2.threshold(gray_u8, 235, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_u8, 80, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    dmg = cv2.bitwise_or(bright, edges)
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return dmg


def run_layer2_damage(
    gray_or_balanced_gray_u8: np.ndarray,
    store: ArtifactStore,
    *,
    contrast_win: int = 31,
    low_contrast_thresh: int = 12,
) -> Layer2Output:
    g = gray_or_balanced_gray_u8

    contrast = _local_contrast_map(g, win=contrast_win)
    uncertainty = (contrast < low_contrast_thresh).astype(np.uint8) * 255
    damage = _detect_holes_and_tears(g)

    uncertainty_ratio = float(np.mean(uncertainty > 0))
    damage_ratio = float(np.mean(damage > 0))
    num_damage_pixels = int(np.sum(damage > 0))

    write_image(store.path("L2", "uncertainty_map.png"), uncertainty)
    write_image(store.path("L2", "damage_mask.png"), damage)
    write_image(store.path("L2", "debug_contrast.png"), safe_uint8(contrast))

    metrics = {
        "uncertainty_ratio": round(uncertainty_ratio, 4),
        "damage_ratio": round(damage_ratio, 4),
        "num_damage_pixels": num_damage_pixels,
        "mean_local_contrast": round(float(np.mean(contrast)), 3),
    }

    debug = {
        "notes": (
            "Layer 2 estimates damage, unreadability, and local uncertainty. "
            "These signals contribute mainly to geometry reliability G(p) and downstream caution."
        )
    }

    return Layer2Output(
        damage_mask_u8=damage,
        uncertainty_u8=uncertainty,
        metrics=metrics,
        debug=debug
    )