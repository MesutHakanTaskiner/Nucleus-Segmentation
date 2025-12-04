import numpy as np
from skimage import morphology


def get_foreground_markers(distance_map_norm: np.ndarray, threshold_ratio: float = 0.35) -> np.ndarray:
    fg = (distance_map_norm > threshold_ratio).astype(np.uint8)
    fg = morphology.remove_small_objects(fg.astype(bool), min_size=10)
    return (fg * 255).astype(np.uint8)
