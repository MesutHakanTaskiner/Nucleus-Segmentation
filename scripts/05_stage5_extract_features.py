from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import read_image_bgr, build_binary_and_instance_masks
from src.metrics import iou_dice
from src.stage2_threshold import run_stage2_threshold
from src.stage3_morphology import run_stage3_morphology
from src.stage4_watershed import run_stage4_watershed
from src.stage5_features import run_stage5_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage5"))

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
    mask_paths = [Path(p) for p in str(row["mask_paths"]).split(";") if p.strip()]

    img = read_image_bgr(image_path)
    gt_merged01, gt_inst = build_binary_and_instance_masks(mask_paths)
    gt_count = int(gt_inst.max())

    out = args.out_dir / image_id
    out.mkdir(parents=True, exist_ok=True)

    # Stage 2
    s2 = run_stage2_threshold(img, blur_ksize=args.blur_ksize)

    # Stage 3
    s3 = run_stage3_morphology(
        s2.pred_mask01,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        min_area=args.min_area,
    )

    # Stage 4
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
    pred_merged01 = (pred_labels > 0).astype(np.uint8)

    # Stage 5
    s5 = run_stage5_features(img, pred_labels)
    pred_count = int(s5.count)

    # Binary metrics vs GT merged (sanity)
    iou, dice = iou_dice(pred_merged01, gt_merged01)

    # Save features CSV
    features_path = out / "nuclei_features.csv"
    s5.features.to_csv(features_path, index=False)

    # Summary
    report = {
        "image_id": image_id,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_error": int(pred_count - gt_count),
        "binary_metrics_vs_gt_merged": {"iou": float(iou), "dice": float(dice)},
        "stage4_params": {
            "dist_erode_iter": int(args.dist_erode_iter),
            "min_distance": int(args.min_distance),
            "peak_rel_thresh": float(args.peak_rel_thresh),
            "seed_dilate": int(args.seed_dilate),
            "bg_dilate_iter": int(args.bg_dilate_iter),
            "min_instance_area": int(args.min_instance_area),
        },
        "features_csv": str(features_path),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
