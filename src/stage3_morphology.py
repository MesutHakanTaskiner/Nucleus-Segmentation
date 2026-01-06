from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class Stage3Output:
    opened01: np.ndarray
    closed01: np.ndarray
    filled01: np.ndarray
    filtered01: np.ndarray


def _binary01(mask01: np.ndarray) -> np.ndarray:
    return (mask01 > 0).astype(np.uint8)


def morph_open(mask01: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Opening = erosion then dilation.
    Removes small bright noise and thin protrusions.
    """
    m = _binary01(mask01) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return (out > 0).astype(np.uint8)


def morph_close(mask01: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Closing = dilation then erosion.
    Closes small holes/gaps and connects nearby regions.
    """
    m = _binary01(mask01) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return (out > 0).astype(np.uint8)


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground using flood fill from the border.
    Steps:
    - Flood fill background from (0,0) on the inverted mask
    - Invert flood-filled result to get filled holes
    """
    m = _binary01(mask01)
    inv = (1 - m).astype(np.uint8) * 255

    h, w = inv.shape[:2]
    flood = inv.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, mask, (0, 0), 255)

    # flood now has background connected to border filled with 255
    # holes remain 0 in flood; invert to get holes as 255
    holes = cv2.bitwise_not(flood)

    filled = cv2.bitwise_or(m * 255, holes)
    return (filled > 0).astype(np.uint8)


def remove_small_components(mask01: np.ndarray, min_area: int = 30) -> np.ndarray:
    """
    Remove connected components smaller than min_area (in pixels).
    """
    m = _binary01(mask01)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)

    # stats: [label, x, y, w, h, area] for each component; component 0 is background
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lab] = 1
    return out


def run_stage3_morphology(
    pred01: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 3,
    min_area: int = 30,
) -> Stage3Output:
    """
    Stage 3: opening -> closing -> hole filling -> small component removal.
    """
    opened = morph_open(pred01, ksize=open_ksize, iterations=1)
    closed = morph_close(opened, ksize=close_ksize, iterations=1)
    filled = fill_holes(closed)
    filtered = remove_small_components(filled, min_area=min_area)
    return Stage3Output(opened01=opened, closed01=closed, filled01=filled, filtered01=filtered)
