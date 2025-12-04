from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk with unchanged channels."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    print(f"Loaded image {path} with shape {image.shape} and dtype {image.dtype}")
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
