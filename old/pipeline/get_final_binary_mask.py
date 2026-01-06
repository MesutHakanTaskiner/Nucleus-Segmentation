import numpy as np


def get_final_binary_mask(label_map: np.ndarray) -> np.ndarray:
    return (label_map > 0).astype(np.uint8) * 255
