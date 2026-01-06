import random

import cv2
import numpy as np
from skimage import color, util


def overlay_labels_on_image(image_bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    rand_seed = 42
    random.seed(rand_seed)
    max_label = labels.max()
    if image_bgr.ndim == 2:
        base_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.shape[2] == 3:
        base_rgb = image_bgr[..., ::-1]
    else:
        base_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    if max_label == 0:
        overlay_rgb = base_rgb
    else:
        colors = np.array([[random.random(), random.random(), random.random()] for _ in range(max_label + 1)])
        colors[0] = [0, 0, 0]
        overlay_rgb = color.label2rgb(labels, image=base_rgb, colors=colors.tolist(), alpha=alpha, bg_label=0)
        overlay_rgb = util.img_as_ubyte(overlay_rgb)
    overlay_bgr = overlay_rgb[..., ::-1]
    return overlay_bgr
