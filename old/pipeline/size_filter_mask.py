import numpy as np
from skimage import measure


def size_filter_mask(mask_binary: np.ndarray, min_area: int = 20, max_area: int | None = None) -> np.ndarray:
    labeled = measure.label(mask_binary > 0, connectivity=2)
    props = measure.regionprops(labeled)
    keep = np.zeros_like(mask_binary, dtype=np.uint8)
    for prop in props:
        area = prop.area
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        keep[labeled == prop.label] = 255
    return keep
