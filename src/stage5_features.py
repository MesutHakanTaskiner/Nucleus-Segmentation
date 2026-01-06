from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops


@dataclass(frozen=True)
class Stage5Output:
    features: pd.DataFrame
    count: int


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


def circularity(area: float, perimeter: float) -> float:
    """
    Circularity = 4*pi*Area / (Perimeter^2)
    1.0 is a perfect circle (in continuous domain). Discrete pixel effects apply.
    """
    if perimeter <= 1e-8:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def extract_features(labels: np.ndarray, gray_u8: np.ndarray) -> pd.DataFrame:
    """
    Extract per-instance features from an instance label mask and a grayscale intensity image.
    labels: int32, 0 background, 1..N objects
    gray_u8: uint8 grayscale image (same H,W)
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D")
    if gray_u8.ndim != 2:
        raise ValueError("gray_u8 must be 2D")
    if labels.shape != gray_u8.shape:
        raise ValueError("labels and gray_u8 must have the same shape")

    props = regionprops(labels.astype(np.int32), intensity_image=gray_u8.astype(np.uint8))

    rows = []
    for r in props:
        # regionprops uses (row, col) for centroid
        cy, cx = r.centroid
        area = float(r.area)
        perim = float(r.perimeter)  # approximate perimeter in pixels

        rows.append(
            {
                "label_id": int(r.label),
                "area": area,
                "perimeter": perim,
                "circularity": circularity(area, perim),
                "major_axis_length": float(r.major_axis_length),
                "minor_axis_length": float(r.minor_axis_length),
                "eccentricity": float(r.eccentricity),
                "solidity": float(r.solidity),
                "extent": float(r.extent),
                "convex_area": float(r.convex_area),
                "bbox_min_row": int(r.bbox[0]),
                "bbox_min_col": int(r.bbox[1]),
                "bbox_max_row": int(r.bbox[2]),
                "bbox_max_col": int(r.bbox[3]),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "mean_intensity": float(r.mean_intensity),
            }
        )

    df = pd.DataFrame(rows).sort_values("label_id").reset_index(drop=True)
    return df


def run_stage5_features(image_bgr: np.ndarray, labels: np.ndarray) -> Stage5Output:
    """
    Stage 5: compute per-nucleus features and nucleus count from labels.
    """
    gray = to_gray_uint8(image_bgr)
    df = extract_features(labels=labels, gray_u8=gray)
    count = int(df.shape[0])
    return Stage5Output(features=df, count=count)
