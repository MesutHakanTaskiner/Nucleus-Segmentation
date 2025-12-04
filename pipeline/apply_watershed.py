import numpy as np
from skimage import segmentation


def apply_watershed(gradient_image: np.ndarray, marker_image: np.ndarray) -> np.ndarray:
    return segmentation.watershed(gradient_image, markers=marker_image, mask=marker_image > 0)
