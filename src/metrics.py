from __future__ import annotations

import numpy as np


def iou_dice(pred01: np.ndarray, gt01: np.ndarray) -> tuple[float, float]:
    """
    Compute IoU and Dice for binary masks (0/1).
    """
    p = (pred01 > 0).astype(np.uint8)
    g = (gt01 > 0).astype(np.uint8)

    inter = int(np.sum((p == 1) & (g == 1)))
    union = int(np.sum((p == 1) | (g == 1)))
    p_sum = int(np.sum(p))
    g_sum = int(np.sum(g))

    iou = float(inter / union) if union > 0 else 1.0
    dice = float((2 * inter) / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 1.0
    return iou, dice
