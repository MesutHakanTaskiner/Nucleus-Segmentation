import cv2
import numpy as np


def segment_adaptive(image_gray_enhanced: np.ndarray, block_size: int = 35, c: int = 2, invert: bool = False) -> np.ndarray:
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    adaptive = cv2.adaptiveThreshold(
        image_gray_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return adaptive
