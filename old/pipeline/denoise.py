import cv2
import numpy as np


def denoise(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = (ksize, ksize)
    return cv2.GaussianBlur(image_gray, k, 0)
