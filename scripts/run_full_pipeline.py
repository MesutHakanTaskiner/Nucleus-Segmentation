from __future__ import annotations

"""
Run stages 2–6 for all samples in a manifest.
Saves final artifacts (labels, overlays, comparison) and per-sample feature CSVs.
"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import read_image_bgr, build_binary_and_instance_masks
from src.metrics import iou_dice
from src.stage2_threshold import run_stage2_threshold
from src.stage3_morphology import run_stage3_morphology
from src.stage4_watershed import run_stage4_watershed
from src.stage5_features import run_stage5_features
from src.stage6_final_save import save_final_artifacts


def process_sample(
    image_id: str,
    image_path: Path,
    mask_paths: list[Path],
    out_root: Path,
    s2_blur_ksize: int,
    s3_open_ksize: int,
    s3_close_ksize: int,
    s3_min_area: int,
    s4_dist_erode_iter: int,
    s4_min_distance: int,
    s4_peak_rel_thresh: float,
    s4_seed_dilate: int,
    s4_bg_dilate_iter: int,
    s4_min_instance_area: int,
) -> dict:
    img = read_image_bgr(image_path)
    gt_bin, gt_inst = build_binary_and_instance_masks(mask_paths)

    # Stages 2–4: prediction
    s2 = run_stage2_threshold(img, blur_ksize=s2_blur_ksize)
    s3 = run_stage3_morphology(
        s2.pred_mask01,
        open_ksize=s3_open_ksize,
        close_ksize=s3_close_ksize,
        min_area=s3_min_area,
    )
    s4 = run_stage4_watershed(
        image_bgr=img,
        mask01=s3.filtered01,
        dist_erode_iter=s4_dist_erode_iter,
        min_distance=s4_min_distance,
        peak_rel_thresh=s4_peak_rel_thresh,
        seed_dilate=s4_seed_dilate,
        bg_dilate_iter=s4_bg_dilate_iter,
        min_instance_area=s4_min_instance_area,
    )

    pred_labels = s4.labels
    pred_bin01 = (pred_labels > 0).astype(np.uint8)

    # Stage 5: features
    s5 = run_stage5_features(img, pred_labels)

    # Metrics
    iou, dice = iou_dice(pred_bin01, gt_bin)

    # Save artifacts
    out_dir = out_root / image_id
    artifacts = save_final_artifacts(
        out_dir=out_dir,
        image_bgr=img,
        pred_labels=pred_labels,
        gt_mask_paths=mask_paths,
    )

    features_path = out_dir / "nuclei_features.csv"
    s5.features.to_csv(features_path, index=False)

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "num_gt_instances": int(gt_inst.max()),
        "num_pred_instances": int(pred_labels.max()),
        "count_error": int(pred_labels.max() - gt_inst.max()),
        "binary_metrics_vs_gt": {"iou": float(iou), "dice": float(dice)},
        "saved": {
            "pred_labels": str(artifacts.pred_labels_path),
            "gt_labels": str(artifacts.gt_labels_path),
            "comparison": str(artifacts.comparison_path),
            "features_csv": str(features_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/full_run"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples")

    # Stage 2
    parser.add_argument("--blur-ksize", type=int, default=5)
    # Stage 3
    parser.add_argument("--open-ksize", type=int, default=3)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=30)
    # Stage 4
    parser.add_argument("--dist-erode-iter", type=int, default=1)
    parser.add_argument("--min-distance", type=int, default=6)
    parser.add_argument("--peak-rel-thresh", type=float, default=0.15)
    parser.add_argument("--seed-dilate", type=int, default=2)
    parser.add_argument("--bg-dilate-iter", type=int, default=2)
    parser.add_argument("--min-instance-area", type=int, default=20)

    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.iloc[: args.limit]

    results = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        image_path = Path(row["image_path"])
        mask_paths = [Path(p) for p in str(row["mask_paths"]).split(";") if p.strip()]

        try:
            summary = process_sample(
                image_id=image_id,
                image_path=image_path,
                mask_paths=mask_paths,
                out_root=args.out_dir,
                s2_blur_ksize=args.blur_ksize,
                s3_open_ksize=args.open_ksize,
                s3_close_ksize=args.close_ksize,
                s3_min_area=args.min_area,
                s4_dist_erode_iter=args.dist_erode_iter,
                s4_min_distance=args.min_distance,
                s4_peak_rel_thresh=args.peak_rel_thresh,
                s4_seed_dilate=args.seed_dilate,
                s4_bg_dilate_iter=args.bg_dilate_iter,
                s4_min_instance_area=args.min_instance_area,
            )
            results.append(summary)
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Error processing sample {image_id}: {e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} sample(s). Summary saved to {args.out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
