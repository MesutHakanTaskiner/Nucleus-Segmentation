# Nucleus Segmentation Pipeline

End-to-end nucleus segmentation workflow built around a small set of reusable stages. The repository contains reusable functions under `src/` and runnable scripts under `scripts/` that chain the stages together. The legacy prototype lives in `old/` and is not used by the current pipeline.

## Dependencies
- Python 3.9+
- numpy, pandas
- opencv-python
- scikit-image

Install inside a virtualenv/conda env:
```bash
pip install numpy pandas opencv-python scikit-image
```

> Run scripts from the repository root so imports like `from src...` resolve correctly.

## Data layout & manifest
Supported layouts are detected automatically:
- **DSB-style:** `<id>/images/<id>.png` and `<id>/masks/*.png`
- **Flat:** `images/*.png` and `masks/*.png` with matching stems

Create a manifest CSV the rest of the pipeline can consume:
```bash
python scripts/00_make_manifest.py \
  --data-root data/raw/U_NET/test \
  --out data/manifest.csv
```

`data/manifest.csv` fields:
- `image_id`: unique id (folder name or stem)
- `image_path`: absolute/relative image path
- `mask_paths`: semicolon-separated list of mask paths
- `num_masks`: count of masks for the sample

## Pipeline scripts
Each script takes `--manifest` and `--idx` to pick a row from the manifest.

- **01_inspect_sample.py** – quick sanity check; saves the image, merged mask, instance visualization, and overlays.
  ```bash
  python scripts/01_inspect_sample.py --manifest data/manifest.csv --idx 0 --out-dir results/inspect
  ```

- **02_stage2_threshold_baseline.py** – grayscale + Gaussian blur + Otsu threshold with auto polarity; reports IoU/Dice vs. GT merged mask.
  ```bash
  python scripts/02_stage2_threshold_baseline.py --manifest data/manifest.csv --idx 0 --blur-ksize 5 --out-dir results/stage2
  ```

- **03_stage3_morph_cleanup.py** – runs stage 2 then morphology (open/close/hole-fill/small-component removal); reports metrics and saves overlays.
  ```bash
  python scripts/03_stage3_morph_cleanup.py --manifest data/manifest.csv --idx 0 \
    --open-ksize 3 --close-ksize 3 --min-area 30 --out-dir results/stage3
  ```

- **04_stage4_watershed_instances.py** – stand-alone watershed experiment tool (distance transform + peak-local-max seeds). Useful for tuning stage 4 parameters.

- **05_stage5_extract_features.py** – runs stages 2–4, then extracts per-nucleus features and writes `nuclei_features.csv` plus a summary JSON.
  ```bash
  python scripts/05_stage5_extract_features.py --manifest data/manifest.csv --idx 0 \
    --blur-ksize 5 \
    --open-ksize 3 --close-ksize 3 --min-area 30 \
    --dist-erode-iter 1 --min-distance 6 --peak-rel-thresh 0.15 --seed-dilate 2 \
    --bg-dilate-iter 2 --min-instance-area 20 \
    --out-dir results/stage5
  ```

- **06_stage6_final_outputs.py** – full run producing final 16-bit label PNGs, binaries, overlays, and a comparison panel plus metrics.
  ```bash
  python scripts/06_stage6_final_outputs.py --manifest data/manifest.csv --idx 0 \
    --blur-ksize 5 \
    --open-ksize 3 --close-ksize 3 --min-area 30 \
    --dist-erode-iter 1 --min-distance 6 --peak-rel-thresh 0.15 --seed-dilate 2 \
    --bg-dilate-iter 2 --min-instance-area 20 \
    --out-dir results/final
  ```

- **run_full_pipeline.py** – processes the entire manifest (or first N samples) through stages 2–6, saving final artifacts and per-sample feature CSVs plus a consolidated summary.
  ```bash
  python scripts/run_full_pipeline.py --manifest data/manifest.csv --out-dir results/full_run --limit 10 \
    --blur-ksize 5 \
    --open-ksize 3 --close-ksize 3 --min-area 30 \
    --dist-erode-iter 1 --min-distance 6 --peak-rel-thresh 0.15 --seed-dilate 2 \
    --bg-dilate-iter 2 --min-instance-area 20
  ```

## Core modules (`src/`)
- `dataset.py` – dataset discovery, manifest writing, and image/mask loaders.
- `metrics.py` – IoU/Dice for binary masks.
- `stage2_threshold.py` – grayscale + blur + Otsu threshold (auto polarity).
- `stage3_morphology.py` – opening, closing, hole filling, and small-component filtering.
- `stage4_watershed.py` – distance-transform + peak-local-max seeding + watershed + post-filtering.
- `stage5_features.py` – per-instance feature extraction using `skimage.regionprops`.
- `stage6_final_save.py` – saves predicted/GT labels (16-bit), binaries, overlays, and comparison panel.
- `viz.py` – overlay and PNG save helpers.

## Tips
- Keep resolutions consistent across images/masks; mismatched shapes will raise errors.
- Use `--idx` to iterate through samples when tuning parameters.
- Results are written under `results/` by default; adjust `--out-dir` to change destinations.
