from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def overlay_mask_on_image(image_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create a simple overlay visualization:
    - mask pixels are tinted red
    - alpha controls blending
    """
    if mask01.ndim != 2:
        raise ValueError("mask01 must be a 2D array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a BGR color image")

    mask = (mask01 > 0).astype(np.uint8)
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)  # red in BGR
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
