from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class Stage4Output:
    dist_u8: np.ndarray
    seeds01: np.ndarray
    sure_bg01: np.ndarray
    unknown01: np.ndarray
    markers_pre: np.ndarray
    labels: np.ndarray


def _binary01(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


def normalize_to_u8(x: np.ndarray) -> np.ndarray:
    """
    Normalize float array to 0..255 uint8 for visualization.
    """
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn + 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    return (y * 255.0).astype(np.uint8)


def compute_distance_map(mask01: np.ndarray, erode_iter: int = 0) -> np.ndarray:
    """
    Distance transform on binary mask. Optional erosion can help split touching objects,
    but erosion can also delete small nuclei. Start with 0.
    """
    m = _binary01(mask01)
    if erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(m, k, iterations=erode_iter)
    dist = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5)
    return dist.astype(np.float32)


def build_markers_peak_local_max(
    mask01: np.ndarray,
    dist: np.ndarray,
    min_distance: int = 6,
    peak_rel_thresh: float = 0.15,
    seed_dilate: int = 2,
    bg_dilate_iter: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build watershed markers using local maxima on distance map.

    - peak_local_max finds peak coordinates in dist within mask.
    - We dilate seed points into tiny blobs -> connectedComponents -> unique markers.
    - Background is 1, seeds are 2..K+1, unknown is 0 (cv2.watershed convention).
    """
    m = _binary01(mask01)

    dmax = float(np.max(dist))
    if dmax <= 1e-8:
        seeds = np.zeros_like(m, dtype=np.uint8)
    else:
        coords = peak_local_max(
            dist,
            labels=m,
            min_distance=max(1, int(min_distance)),
            threshold_abs=peak_rel_thresh * dmax,
            exclude_border=False,
        )
        seeds = np.zeros_like(m, dtype=np.uint8)
        if coords.size > 0:
            seeds[coords[:, 0], coords[:, 1]] = 1

        if seed_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed_dilate + 1, 2 * seed_dilate + 1))
            seeds = cv2.dilate(seeds, k, iterations=1)

    kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(m, kbg, iterations=int(bg_dilate_iter))

    unknown = ((sure_bg > 0) & (seeds == 0)).astype(np.uint8)

    num, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    markers = seed_labels.astype(np.int32) + 1
    markers[unknown > 0] = 0

    return seeds, sure_bg, unknown, markers


def watershed_input_gradient(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a cleaner watershed input image from gradient magnitude.
    cv2.watershed expects 3-channel 8-bit or 32-bit image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag_u8 = normalize_to_u8(mag)
    return cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    """
    Make labels contiguous 1..N (no gaps).
    """
    lab = labels.astype(np.int32, copy=False)
    ids = np.unique(lab)
    ids = ids[ids != 0]
    mapping = {int(old): i + 1 for i, old in enumerate(ids)}
    out = np.zeros_like(lab, dtype=np.int32)
    for old, new in mapping.items():
        out[lab == old] = new
    return out


def filter_instances_by_area(labels: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Remove tiny predicted instances (common cause of overcount).
    """
    lab = labels.astype(np.int32, copy=False)
    if lab.max() == 0:
        return lab.copy()
    areas = np.bincount(lab.ravel())
    out = lab.copy()
    for i in range(1, len(areas)):
        if areas[i] < min_area:
            out[out == i] = 0
    return relabel_compact(out)


def run_stage4_watershed(
    image_bgr: np.ndarray,
    mask01: np.ndarray,
    dist_erode_iter: int = 0,
    min_distance: int = 6,
    peak_rel_thresh: float = 0.15,
    seed_dilate: int = 2,
    bg_dilate_iter: int = 2,
    min_instance_area: int = 20,
) -> Stage4Output:
    """
    Stage 4 (improved):
    - distance transform
    - peak_local_max seeds
    - watershed on gradient image
    - remove tiny instances
    """
    dist = compute_distance_map(mask01, erode_iter=dist_erode_iter)
    dist_u8 = normalize_to_u8(dist)

    seeds01, sure_bg01, unknown01, markers = build_markers_peak_local_max(
        mask01=mask01,
        dist=dist,
        min_distance=min_distance,
        peak_rel_thresh=peak_rel_thresh,
        seed_dilate=seed_dilate,
        bg_dilate_iter=bg_dilate_iter,
    )

    ws_img = watershed_input_gradient(image_bgr)

    markers_ws = markers.copy()
    cv2.watershed(ws_img, markers_ws)

    labels = np.zeros_like(markers_ws, dtype=np.int32)
    labels[markers_ws >= 2] = markers_ws[markers_ws >= 2] - 1
    labels = filter_instances_by_area(labels, min_area=int(min_instance_area))

    return Stage4Output(
        dist_u8=dist_u8,
        seeds01=seeds01,
        sure_bg01=sure_bg01,
        unknown01=unknown01,
        markers_pre=markers,
        labels=labels,
    )
