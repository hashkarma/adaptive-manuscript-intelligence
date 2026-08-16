from __future__ import annotations
import os
import cv2
import numpy as np


def read_image_bgr(path: str) -> np.ndarray:
    """
    Reads an image from disk as BGR (OpenCV default).

    Why BGR?
    - OpenCV reads images in BGR order.
    - We keep internal processing consistent, and convert only when needed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read the image (corrupt/unsupported): {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    """
    Saves an image to disk. Creates parent folders if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image to: {path}")


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def safe_uint8(img: np.ndarray) -> np.ndarray:
    """
    Ensures the array is uint8 in [0,255].
    """
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)