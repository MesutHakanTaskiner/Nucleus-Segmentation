# Nucleus Segmentation with Classical Computer Vision

Lightweight watershed-based nucleus segmentation on histopathology slides using only classical computer-vision steps (no deep learning). The pipeline produces overlays, intermediate masks, per-nucleus features (including counts), and optional quality metrics against ground-truth masks.

## Requirements
- Python 3.10+ (virtual environment recommended)
- Packages: `opencv-python`, `scikit-image`, `numpy`, `pandas`

Install into your environment:
```
python -m pip install -U pip
pip install opencv-python scikit-image numpy pandas
```

## Data layout
Place images under `data/`. Ground-truth masks are optional but enable IoU/Dice/PSNR/SSIM reporting.
```
data/
  MoNuSeg/
    img/    # input images
    label/  # ground-truth masks (same filenames as images)
  kmms/
    images/ # input images
    masks/  # ground-truth masks (same filenames as images)
```
During a run the code will look for a mask named like the input image either at `data/masks_gt/<image>.png` or beside the image in a sibling `masks/` folder.

## Run the pipeline
- Use the first image found under `data/`:
```
python main_nucleus_segmentation.py --results-dir results/sample_run
```
- Or specify an image explicitly:
```
python main_nucleus_segmentation.py --input data/MoNuSeg/img/0001.png --results-dir results/0001
```
- Process all images under a directory (each image gets its own subfolder named after the file stem):
```
python main_nucleus_segmentation.py --input-dir data/MoNuSeg/img --results-dir results/monuseg_all
```
  - Optional: `--limit 5` to process only the first 5 images.

Outputs (under the chosen `results` directory):
- `overlays/`: original, grayscale, denoised, CLAHE, distance map visualization, markers visualization, gradient, watershed label overlay.
- `masks/`: Otsu mask, cleaned mask, size-filtered mask, saved foreground/background markers, final binary mask.
- `features/`:
  - `nuclei_features_example.csv`: one row per nucleus (area/perimeter/axes/eccentricity/circularity); the nucleus count is the number of rows.
  - `psnr_ssim_examples.csv`: PSNR/SSIM of masks vs ground truth (written when a GT mask is found).
  - `metrics_examples.csv`: IoU and Dice vs ground truth (written when a GT mask is found).

Quick way to print the nucleus count after a run:
```
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/0001/features/nuclei_features_example.csv")
print("Nucleus count:", len(df))
PY
```

## Pipeline outline (classical CV only)
1. Load BGR image → grayscale → Gaussian denoise → CLAHE contrast enhancement.
2. Otsu threshold → morphological open/close → size filtering.
3. Distance transform (with a light erosion beforehand) → normalization.
4. Foreground markers from distance peaks (threshold sweep + h-max option); background markers by dilating the cleaned mask and inverting.
5. Sobel gradient (smoothed) + watershed with the combined markers.
6. Relabel, build final binary mask, extract per-nucleus features, and render overlays.
7. Optional evaluation: PSNR/SSIM and IoU/Dice when a ground-truth mask is available.

## Tips
- Tune parameters in `pipeline/process_image.py` (e.g., `min_marker_area`, `threshold_ratio`, `footprint_size`, `dist_pre_erosion_iter`) to better separate touching nuclei.
- Images are expected as 8-bit grayscale or BGR; convert before running if needed.
- For batch processing, `pipeline/process_directory.py` shows how to hook the same pipeline into a directory traversal (currently it processes the first found image; extend the loop to process all).
