import numpy as np
from skimage import exposure


def visualize_distance_map(distance_map_norm: np.ndarray) -> np.ndarray:
    vis = exposure.rescale_intensity(distance_map_norm, out_range=(0, 255))
    return vis.astype(np.uint8)
