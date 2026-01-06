from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from src.viz import overlay_mask_on_image, save_png


@dataclass(frozen=True)
class FinalArtifacts:
    pred_labels_path: Path
    gt_labels_path: Path
    pred_binary_path: Path
    gt_binary_path: Path
    comparison_path: Path


def save_labels_png16(path: Path, labels: np.ndarray) -> None:
    """
    Save instance labels as 16-bit PNG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lab16 = labels.astype(np.uint16)
    ok = cv2.imwrite(str(path), lab16)
    if not ok:
        raise RuntimeError(f"Failed to write 16-bit labels PNG: {path}")


def labels_to_colormap(labels: np.ndarray) -> np.ndarray:
    """
    Convert labels to a colorful visualization (uint8 BGR).
    """
    lab = labels.astype(np.float32)
    mx = float(np.max(lab))
    if mx <= 1e-8:
        u8 = np.zeros_like(labels, dtype=np.uint8)
    else:
        u8 = (lab / mx * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def stack_horiz(images: list[np.ndarray]) -> np.ndarray:
    """
    Stack images horizontally after making them same height.
    """
    heights = [im.shape[0] for im in images]
    h = min(heights)
    resized = []
    for im in images:
        if im.shape[0] != h:
            w = int(im.shape[1] * (h / im.shape[0]))
            resized.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST))
        else:
            resized.append(im)
    return np.hstack(resized)


def build_gt_instance_from_masks(gt_mask_paths: list[Path]) -> np.ndarray:
    """
    Build GT instance labels from per-instance masks (DSB2018 style).
    """
    if len(gt_mask_paths) == 0:
        raise ValueError("gt_mask_paths is empty")
    # Read first to get shape
    m0 = cv2.imread(str(gt_mask_paths[0]), cv2.IMREAD_UNCHANGED)
    if m0 is None:
        raise RuntimeError(f"Failed to read GT mask: {gt_mask_paths[0]}")
    if m0.ndim == 3:
        m0 = m0[..., 0]
    h, w = m0.shape[:2]

    inst = np.zeros((h, w), dtype=np.int32)
    for i, p in enumerate(gt_mask_paths, start=1):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f"Failed to read GT mask: {p}")
        if m.ndim == 3:
            m = m[..., 0]
        m01 = (m > 0).astype(np.uint8)
        inst[m01 > 0] = i
    return inst


def save_final_artifacts(
    out_dir: Path,
    image_bgr: np.ndarray,
    pred_labels: np.ndarray,
    gt_mask_paths: list[Path],
) -> FinalArtifacts:
    """
    Save final prediction/GT artifacts and a comparison panel.

    - pred_labels, gt_labels are saved as 16-bit PNG
    - binary masks and overlays are saved for quick visual inspection
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT instance labels
    gt_labels = build_gt_instance_from_masks(gt_mask_paths)

    # Binary masks
    pred_bin01 = (pred_labels > 0).astype(np.uint8)
    gt_bin01 = (gt_labels > 0).astype(np.uint8)

    # Save label PNGs (16-bit)
    pred_labels_path = out_dir / "pred_labels.png"
    gt_labels_path = out_dir / "gt_labels.png"
    save_labels_png16(pred_labels_path, pred_labels)
    save_labels_png16(gt_labels_path, gt_labels)

    # Save binary PNGs
    pred_binary_path = out_dir / "pred_binary.png"
    gt_binary_path = out_dir / "gt_binary.png"
    save_png(pred_binary_path, (pred_bin01 * 255).astype(np.uint8))
    save_png(gt_binary_path, (gt_bin01 * 255).astype(np.uint8))

    # Visualizations
    pred_color = labels_to_colormap(pred_labels)
    gt_color = labels_to_colormap(gt_labels)

    overlay_pred = overlay_mask_on_image(image_bgr, pred_bin01, alpha=0.45)
    overlay_gt = overlay_mask_on_image(image_bgr, gt_bin01, alpha=0.45)

    save_png(out_dir / "pred_color.png", pred_color)
    save_png(out_dir / "gt_color.png", gt_color)
    save_png(out_dir / "overlay_pred.png", overlay_pred)
    save_png(out_dir / "overlay_gt.png", overlay_gt)

    # Build a single comparison panel
    panel = stack_horiz([image_bgr, overlay_gt, overlay_pred, gt_color, pred_color])
    comparison_path = out_dir / "comparison.png"
    save_png(comparison_path, panel)

    return FinalArtifacts(
        pred_labels_path=pred_labels_path,
        gt_labels_path=gt_labels_path,
        pred_binary_path=pred_binary_path,
        gt_binary_path=gt_binary_path,
        comparison_path=comparison_path,
    )
