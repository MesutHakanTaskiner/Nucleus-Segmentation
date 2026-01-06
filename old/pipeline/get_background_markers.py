import cv2
import numpy as np


def get_background_markers(mask_clean: np.ndarray, dilation_iter: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask_clean, kernel, iterations=dilation_iter)
    bg = cv2.bitwise_not(sure_bg)
    return bg
