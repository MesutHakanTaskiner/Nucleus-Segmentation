import numpy as np
from skimage import color, metrics


def _to_gray_float01(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/GRAY/4-channel image to grayscale float [0,1]."""
    img = image
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 3:
        gray = color.rgb2gray(img[..., ::-1])  # assume BGR -> RGB
    else:
        # assume 4-channel BGRA
        gray = color.rgb2gray(img[..., :3][..., ::-1])
    gray = gray.astype(np.float64)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def compute_psnr_ssim_arrays(image_a: np.ndarray, image_b: np.ndarray) -> tuple[float, float]:
    a_gray = _to_gray_float01(image_a)
    b_gray = _to_gray_float01(image_b)
    psnr = metrics.peak_signal_noise_ratio(a_gray, b_gray, data_range=1.0)
    ssim = metrics.structural_similarity(a_gray, b_gray, data_range=1.0)
    return psnr, ssim
