from pathlib import Path

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


def process_image(image_path: Path, results_dir: Path) -> None:
    """Run the full classical pipeline on one image and write all intermediates/results."""
    overlays_dir = results_dir / "overlays"
    masks_dir = results_dir / "masks"
    features_dir = results_dir / "features"

    # Load and stash original
    image_bgr = load_image(image_path)
    save_image(overlays_dir / "original_example.png", image_bgr)

    # Preprocess: grayscale -> denoise -> CLAHE
    gray = to_grayscale(image_bgr)
    save_image(overlays_dir / "gray_example.png", gray)

    gray_denoised = denoise(gray, ksize=3)
    save_image(overlays_dir / "gray_denoised_example.png", gray_denoised)

    gray_enhanced = enhance_contrast_clahe(gray_denoised)
    save_image(overlays_dir / "gray_enhanced_example.png", gray_enhanced)

    mask_raw = segment_otsu(gray_enhanced, invert=False)
    save_image(masks_dir / "mask_otsu_raw_example.png", mask_raw)

    # Morphology + size filtering
    mask_clean = morphological_cleanup(mask_raw, open_kernel=3, close_kernel=3)
    save_image(masks_dir / "mask_clean_example.png", mask_clean)

    mask_clean_sized = size_filter_mask(mask_clean, min_area=20, max_area=None)
    save_image(masks_dir / "mask_clean_sized_example.png", mask_clean_sized)

    # Marker creation via distance transform
    dist_map = compute_distance_transform(mask_clean_sized)
    dist_map_norm = normalize_distance_map(dist_map)
    save_image(overlays_dir / "distance_map_example.png", visualize_distance_map(dist_map_norm))

    fg_markers = get_foreground_markers(dist_map_norm, threshold_ratio=0.35)
    save_image(masks_dir / "markers_fg_example.png", fg_markers)

    bg_markers = get_background_markers(mask_clean_sized, dilation_iter=2)
    save_image(masks_dir / "markers_bg_example.png", bg_markers)

    marker_image = build_marker_image(fg_markers, bg_markers)
    save_image(overlays_dir / "markers_combined_example.png", visualize_markers(marker_image))

    # Gradient + watershed
    gradient_image = compute_gradient(gray_enhanced.astype(np.float32))
    grad_vis = exposure.rescale_intensity(gradient_image, out_range=(0, 255)).astype(np.uint8)
    save_image(overlays_dir / "gradient_example.png", grad_vis)

    labels_raw = apply_watershed(gradient_image, marker_image)
    labels = postprocess_labels(labels_raw)
    overlay = overlay_labels_on_image(image_bgr, labels, alpha=0.6)
    save_image(overlays_dir / "segmentation_overlay_example.png", overlay)

    # Final mask and features
    mask_final = get_final_binary_mask(labels)
    save_image(masks_dir / "mask_final_example.png", mask_final)

    features = extract_nuclei_features(labels, gray_enhanced)
    ensure_dir(features_dir / "nuclei_features_example.csv")
    features.to_csv(features_dir / "nuclei_features_example.csv", index=False)

    # PSNR/SSIM comparisons against original image
    comparisons = [
        ("mask_otsu_raw_example.png", mask_raw),
        ("mask_clean_example.png", mask_clean),
        ("mask_clean_sized_example.png", mask_clean_sized),
        ("mask_final_example.png", mask_final),
    ]
    compare_psnr_ssim_to_reference(
        reference_image=image_bgr,
        targets=comparisons,
        output_csv=features_dir / "psnr_ssim_examples.csv",
    )
