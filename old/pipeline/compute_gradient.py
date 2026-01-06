import numpy as np
from skimage import filters


def compute_gradient(image_gray_enhanced: np.ndarray) -> np.ndarray:
    gx = filters.sobel_h(image_gray_enhanced)
    gy = filters.sobel_v(image_gray_enhanced)
    grad = np.hypot(gx, gy)
    return grad.astype(np.float32)
