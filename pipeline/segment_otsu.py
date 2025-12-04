import cv2
import numpy as np


def segment_otsu(image_gray_enhanced: np.ndarray, invert: bool = False) -> np.ndarray:
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(image_gray_enhanced, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return mask
