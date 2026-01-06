import numpy as np
from skimage import measure


def postprocess_labels(label_image: np.ndarray) -> np.ndarray:
    label_image = np.where(label_image < 0, 0, label_image)
    relabeled = measure.label(label_image > 0, connectivity=2)
    return relabeled.astype(np.int32)
