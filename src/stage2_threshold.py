from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class Stage2Output:
    gray: np.ndarray          # uint8
    blurred: np.ndarray       # uint8
    pred_mask01: np.ndarray   # uint8 {0,1}
    used_invert: bool
    otsu_thresh: float


def to_gray_uint8(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale uint8.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H,W,3)")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def gaussian_blur(gray_u8: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """
    Suppress high-frequency noise/texture before global thresholding.
    """
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    return cv2.GaussianBlur(gray_u8, (ksize, ksize), sigma)


def otsu_binary(gray_u8: np.ndarray, invert: bool) -> tuple[np.ndarray, float]:
    """
    Otsu threshold on uint8 grayscale.
    Returns (mask01, otsu_threshold).
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    t, m255 = cv2.threshold(gray_u8, 0, 255, thresh_type | cv2.THRESH_OTSU)
    m01 = (m255 > 0).astype(np.uint8)
    return m01, float(t)


def choose_polarity_by_fg_fraction(gray_u8: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """
    Try Otsu with invert=False and invert=True; pick the one with smaller foreground fraction.
    This is a simple heuristic: nuclei usually occupy minority pixels.
    """
    m0, t0 = otsu_binary(gray_u8, invert=False)
    m1, t1 = otsu_binary(gray_u8, invert=True)

    frac0 = float(np.count_nonzero(m0) / m0.size)
    frac1 = float(np.count_nonzero(m1) / m1.size)

    # Avoid degenerate masks if possible
    def deg(frac: float) -> bool:
        return frac < 0.001 or frac > 0.999

    if deg(frac0) and not deg(frac1):
        return m1, True, t1
    if deg(frac1) and not deg(frac0):
        return m0, False, t0

    if frac1 < frac0:
        return m1, True, t1
    return m0, False, t0


def run_stage2_threshold(image_bgr: np.ndarray, blur_ksize: int = 5) -> Stage2Output:
    """
    Stage 2: grayscale -> gaussian blur -> otsu threshold (auto polarity).
    """
    gray = to_gray_uint8(image_bgr)
    blurred = gaussian_blur(gray, ksize=blur_ksize, sigma=0.0)
    pred01, used_invert, otsu_t = choose_polarity_by_fg_fraction(blurred)
    return Stage2Output(
        gray=gray,
        blurred=blurred,
        pred_mask01=pred01,
        used_invert=used_invert,
        otsu_thresh=otsu_t,
    )
