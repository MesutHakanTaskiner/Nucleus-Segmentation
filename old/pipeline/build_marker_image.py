import numpy as np
from skimage import measure


def build_marker_image(fg_markers: np.ndarray, bg_markers: np.ndarray) -> np.ndarray:
    fg_labels = measure.label(fg_markers > 0, connectivity=2)
    marker_image = np.zeros_like(fg_labels, dtype=np.int32)
    marker_image[bg_markers > 0] = 1
    marker_image[fg_labels > 0] = fg_labels[fg_labels > 0] + 1  # shift to keep background as 1
    return marker_image
