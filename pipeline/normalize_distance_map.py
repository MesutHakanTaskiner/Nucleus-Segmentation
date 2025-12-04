import numpy as np


def normalize_distance_map(dist_map: np.ndarray) -> np.ndarray:
    if dist_map.max() == 0:
        return dist_map
    return dist_map / dist_map.max()
