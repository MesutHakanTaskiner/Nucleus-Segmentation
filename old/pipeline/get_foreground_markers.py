import numpy as np
from skimage import morphology
from skimage.morphology import ball, disk, h_maxima


def get_foreground_markers(
    distance_map_norm: np.ndarray,
    threshold_ratio: float = 0.35,
    min_marker_area: int = 10,
    use_local_maxima: bool = False,
    footprint_size: int = 5,
    h_max: float = 0.05,
    apply_closing_kernel: int = 0,
) -> np.ndarray:
    """
    Build foreground markers from a distance map.
    - If use_local_maxima: use h-maxima to find peaks as seeds.
    - Else: simple global threshold on normalized distance.
    - Filter tiny markers and optionally close to merge nearby seeds.
    """
    if use_local_maxima:
        footprint = ball(footprint_size) if distance_map_norm.ndim == 3 else disk(footprint_size)
        peaks = h_maxima(distance_map_norm, h=h_max, footprint=footprint)
        fg = peaks.astype(np.uint8)
    else:
        fg = (distance_map_norm > threshold_ratio).astype(np.uint8)

    # remove tiny markers
    fg = morphology.remove_small_objects(fg.astype(bool), min_size=min_marker_area)

    # optional closing to merge very close seeds
    if apply_closing_kernel and apply_closing_kernel > 1:
        selem = disk(apply_closing_kernel)
        fg = morphology.binary_closing(fg, selem)

    return (fg.astype(np.uint8)) * 255
