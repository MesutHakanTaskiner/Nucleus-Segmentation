from pathlib import Path

import cv2
import numpy as np
from skimage import exposure

from .apply_watershed import apply_watershed
from .build_marker_image import build_marker_image
from .compare_psnr_ssim_to_reference import compare_psnr_ssim_to_reference
from .compute_distance_transform import compute_distance_transform
from .compute_gradient import compute_gradient
from .denoise import denoise
from .enhance_contrast_clahe import enhance_contrast_clahe
from .extract_nuclei_features import extract_nuclei_features
from .get_background_markers import get_background_markers
from .get_final_binary_mask import get_final_binary_mask
from .get_foreground_markers import get_foreground_markers
from .load_image import load_image
from .morphological_cleanup import morphological_cleanup
from .normalize_distance_map import normalize_distance_map
from .overlay_labels_on_image import overlay_labels_on_image
from .postprocess_labels import postprocess_labels
from .save_image import save_image
from .segment_otsu import segment_otsu
from .size_filter_mask import size_filter_mask
from .to_grayscale import to_grayscale
from .visualize_distance_map import visualize_distance_map
from .visualize_markers import visualize_markers
from .ensure_dir import ensure_dir
from .compute_iou_dice import compute_iou_dice


def process_image(image_path: Path, results_dir: Path) -> dict:
    """Run the full classical pipeline on one image and write all intermediates/results."""
    overlays_dir = results_dir / "overlays"
    masks_dir = results_dir / "masks"
    features_dir = results_dir / "features"
    results_summary: dict = {}
    base_name = image_path.stem

    # Try to locate a paired ground-truth mask (optional)
    gt_candidates = [
        Path("data") / "masks_gt" / image_path.name,
        image_path.parent.parent / "masks" / image_path.name,
    ]

    masked_image = None
    for candidate in gt_candidates:
        if candidate.exists():
            try:
                masked_image = load_image(candidate)
            except Exception:
                masked_image = None
            break

    # Load and stash original
    image_bgr = load_image(image_path)
    save_image(overlays_dir / f"{base_name}_original.png", image_bgr)

    # Preprocess: grayscale -> denoise -> CLAHE
    gray = to_grayscale(image_bgr)
    save_image(overlays_dir / f"{base_name}_gray.png", gray)

    gray_denoised = denoise(gray, ksize=3)
    save_image(overlays_dir / f"{base_name}_gray_denoised.png", gray_denoised)

    gray_enhanced = enhance_contrast_clahe(gray_denoised)
    save_image(overlays_dir / f"{base_name}_gray_clahe.png", gray_enhanced)

    mask_raw = segment_otsu(gray_enhanced, invert=False)
    save_image(masks_dir / f"{base_name}_mask_otsu.png", mask_raw)

    # Morphology + size filtering
    mask_clean = morphological_cleanup(mask_raw, open_kernel=3, close_kernel=3)
    save_image(masks_dir / f"{base_name}_mask_clean.png", mask_clean)

    mask_clean_sized = size_filter_mask(mask_clean, min_area=20, max_area=None)
    save_image(masks_dir / f"{base_name}_mask_clean_sized.png", mask_clean_sized)

    # Marker creation via distance transform (with optional erosion to help separate overlaps)
    dist_input = mask_clean_sized.copy()
    dist_pre_erosion_iter = 1  # slight erosion to create gaps between touching nuclei
    if dist_pre_erosion_iter > 0:
        dist_input = cv2.erode(dist_input, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=dist_pre_erosion_iter)
    dist_map = compute_distance_transform(dist_input)
    dist_map_norm = normalize_distance_map(dist_map)
    save_image(overlays_dir / f"{base_name}_distance_map.png", visualize_distance_map(dist_map_norm))

    # Foreground markers: sweep multiple threshold ratios and a local-maxima variant
    marker_closing_kernel = 3
    min_marker_area = 10
    threshold_ratios = [0.25, 0.30, 0.35, 0.40]
    for tr in threshold_ratios:
        fg_tmp = get_foreground_markers(
            dist_map_norm,
            threshold_ratio=tr,
            min_marker_area=min_marker_area,
            use_local_maxima=False,
            apply_closing_kernel=marker_closing_kernel,
        )
        save_image(masks_dir / f"{base_name}_markers_fg_tr_{int(tr*100)}.png", fg_tmp)

    fg_markers = get_foreground_markers(
        dist_map_norm,
        threshold_ratio=0.35,
        min_marker_area=min_marker_area,
        use_local_maxima=True,
        footprint_size=5,
        h_max=0.05,
        apply_closing_kernel=marker_closing_kernel,
    )
    save_image(masks_dir / f"{base_name}_markers_fg.png", fg_markers)

    bg_markers = get_background_markers(mask_clean_sized, dilation_iter=2)
    save_image(masks_dir / f"{base_name}_markers_bg.png", bg_markers)

    marker_image = build_marker_image(fg_markers, bg_markers)
    save_image(overlays_dir / f"{base_name}_markers_combined.png", visualize_markers(marker_image))

    # Gradient + watershed
    gray_for_gradient = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)
    gradient_image = compute_gradient(gray_for_gradient.astype(np.float32))
    gradient_image = cv2.GaussianBlur(gradient_image, (3, 3), 0)  # smooth noisy gradients
    grad_vis = exposure.rescale_intensity(gradient_image, out_range=(0, 255)).astype(np.uint8)
    save_image(overlays_dir / f"{base_name}_gradient.png", grad_vis)

    labels_raw = apply_watershed(gradient_image, marker_image)
    labels = postprocess_labels(labels_raw)
    overlay = overlay_labels_on_image(image_bgr, labels, alpha=0.6)
    save_image(overlays_dir / f"{base_name}_segmentation_overlay.png", overlay)

    # Final mask and features
    mask_final = get_final_binary_mask(labels)
    save_image(masks_dir / f"{base_name}_mask_final.png", mask_final)

    features = extract_nuclei_features(labels, gray_enhanced)
    features_csv = features_dir / f"{base_name}_nuclei_features.csv"
    ensure_dir(features_csv)
    features.to_csv(features_csv, index=False)
    nuclei_count = len(features)
    nuclei_count_file = features_dir / f"{base_name}_nuclei_count.txt"
    with open(nuclei_count_file, "w", encoding="utf-8") as f:
        f.write(f"{nuclei_count}\n")
    results_summary["nuclei_count"] = nuclei_count
    results_summary["features_csv"] = features_csv
    results_summary["nuclei_count_file"] = nuclei_count_file
    results_summary["final_mask"] = masks_dir / f"{base_name}_mask_final.png"
    results_summary["overlay_image"] = overlays_dir / f"{base_name}_segmentation_overlay.png"

    # PSNR/SSIM comparisons against original image
    comparisons = [
        (f"{base_name}_mask_otsu.png", mask_raw),
        (f"{base_name}_mask_clean.png", mask_clean),
        (f"{base_name}_mask_clean_sized.png", mask_clean_sized),
        (f"{base_name}_mask_final.png", mask_final),
    ]
    if masked_image is not None:
        compare_psnr_ssim_to_reference(
            reference_image=masked_image,
            targets=comparisons,
            output_csv=features_dir / f"{base_name}_psnr_ssim.csv",
        )
        results_summary["psnr_ssim_csv"] = features_dir / f"{base_name}_psnr_ssim.csv"

    # IoU/Dice against ground-truth (uses masked_image loaded above if present)
    if masked_image is not None:
        gt = masked_image
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_bin = (gt > 0).astype(np.uint8)
        save_image(masks_dir / f"{base_name}_mask_ground_truth.png", gt_bin * 255)
        pred_bin = (mask_final > 0).astype(np.uint8)
        iou, dice = compute_iou_dice(pred_bin, gt_bin)
        metrics_csv = features_dir / f"{base_name}_metrics.csv"
        ensure_dir(metrics_csv)
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("name,iou,dice\n")
            f.write(f"{image_path.name},{iou},{dice}\n")
        results_summary["metrics_csv"] = metrics_csv
        results_summary["iou"] = iou
        results_summary["dice"] = dice

    return results_summary
