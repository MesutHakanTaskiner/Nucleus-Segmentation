import numpy as np


def _to_bool(mask: np.ndarray) -> np.ndarray:
    """Convert any mask array to boolean."""
    if mask.dtype != bool:
        return mask > 0
    return mask


def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Compute IoU and Dice between binary prediction and ground truth masks."""
    pred = _to_bool(pred_mask)
    gt = _to_bool(gt_mask)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    return iou, dice
