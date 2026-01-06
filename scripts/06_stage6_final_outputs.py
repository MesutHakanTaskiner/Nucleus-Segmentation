from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import read_image_bgr
from src.metrics import iou_dice
from src.stage2_threshold import run_stage2_threshold
from src.stage3_morphology import run_stage3_morphology
from src.stage4_watershed import run_stage4_watershed
from src.stage6_final_save import save_final_artifacts

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/final"))

    # Stage 2
    parser.add_argument("--blur-ksize", type=int, default=5)
    # Stage 3
    parser.add_argument("--open-ksize", type=int, default=3)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=30)
    # Stage 4
    parser.add_argument("--dist-erode-iter", type=int, default=1)
    parser.add_argument("--min-distance", type=int, default=6, help="Min distance between peaks in distance map")
    parser.add_argument("--peak-rel-thresh", type=float, default=0.15, help="Relative peak threshold (0..1)")
    parser.add_argument("--seed-dilate", type=int, default=2, help="Seed dilation radius in pixels")
    parser.add_argument("--bg-dilate-iter", type=int, default=2, help="Background dilation iterations")
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=20,
        help="Minimum area (pixels) for predicted instances after watershed",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.idx < 0 or args.idx >= len(df):
        raise ValueError(f"idx out of range: {args.idx} (0..{len(df)-1})")

    row = df.iloc[args.idx]
    image_id = str(row["image_id"])
    image_path = Path(row["image_path"])
    gt_mask_paths = [Path(p) for p in str(row["mask_paths"]).split(";") if p.strip()]

    img = read_image_bgr(image_path)

    # Stages 2-4 to get pred_labels
    s2 = run_stage2_threshold(img, blur_ksize=args.blur_ksize)
    s3 = run_stage3_morphology(
        s2.pred_mask01,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        min_area=args.min_area,
    )
    s4 = run_stage4_watershed(
        image_bgr=img,
        mask01=s3.filtered01,
        dist_erode_iter=args.dist_erode_iter,
        min_distance=args.min_distance,
        peak_rel_thresh=args.peak_rel_thresh,
        seed_dilate=args.seed_dilate,
        bg_dilate_iter=args.bg_dilate_iter,
        min_instance_area=args.min_instance_area,
    )

    pred_labels = s4.labels
    pred_bin01 = (pred_labels > 0).astype(np.uint8)

    # For binary metrics we need merged GT binary; easiest is rebuild via masks in stage6 helper,
    # but here we compute GT binary quickly from mask paths.
    gt_bin01 = np.zeros_like(pred_bin01, dtype=np.uint8)
    for p in gt_mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f"Failed to read GT mask: {p}")
        if m.ndim == 3:
            m = m[..., 0]
        gt_bin01 = np.maximum(gt_bin01, (m > 0).astype(np.uint8))

    iou, dice = iou_dice(pred_bin01, gt_bin01)

    out = args.out_dir / image_id
    artifacts = save_final_artifacts(
        out_dir=out,
        image_bgr=img,
        pred_labels=pred_labels,
        gt_mask_paths=gt_mask_paths,
    )

    report = {
        "image_id": image_id,
        "image_path": str(image_path),
        "pred_instance_count": int(pred_labels.max()),
        "binary_metrics_vs_gt": {"iou": float(iou), "dice": float(dice)},
        "saved": {
            "pred_labels": str(artifacts.pred_labels_path),
            "gt_labels": str(artifacts.gt_labels_path),
            "comparison": str(artifacts.comparison_path),
        },
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
