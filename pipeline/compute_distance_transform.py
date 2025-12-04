import cv2
import numpy as np


def compute_distance_transform(mask_clean: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask_clean, cv2.DIST_L2, 3)
