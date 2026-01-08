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
from src.viz import overlay_mask_on_image, save_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage4"))

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
    pred_bin01 = (pred_labels > 0).astype(np.uint8)

    iou, dice = iou_dice(pred_bin01, gt_merged01)

    # Save artifacts
    save_png(out / "image.png", img)
    save_png(out / "gt_mask_merged.png", (gt_merged01 * 255).astype(np.uint8))
    save_png(out / "pred_stage3_filtered.png", (s3.filtered01 * 255).astype(np.uint8))
    save_png(out / "pred_stage4_labels.png", pred_labels.astype(np.uint16))
    save_png(out / "pred_stage4_binary.png", (pred_bin01 * 255).astype(np.uint8))
    save_png(out / "dist_u8.png", s4.dist_u8)
    save_png(out / "seeds.png", (s4.seeds01 * 255).astype(np.uint8))
    save_png(out / "sure_bg.png", (s4.sure_bg01 * 255).astype(np.uint8))
    save_png(out / "unknown.png", (s4.unknown01 * 255).astype(np.uint8))
    save_png(out / "markers_pre.png", s4.markers_pre.astype(np.uint16))
    save_png(out / "overlay_pred.png", overlay_mask_on_image(img, pred_bin01, alpha=0.45))
    save_png(out / "overlay_gt.png", overlay_mask_on_image(img, gt_merged01, alpha=0.45))

    report = {
        "image_id": image_id,
        "num_gt_instances": int(gt_inst.max()),
        "pred_instance_count": int(pred_labels.max()),
        "binary_metrics_vs_gt_merged": {"iou": float(iou), "dice": float(dice)},
        "stage4_params": {
            "dist_erode_iter": int(args.dist_erode_iter),
            "min_distance": int(args.min_distance),
            "peak_rel_thresh": float(args.peak_rel_thresh),
            "seed_dilate": int(args.seed_dilate),
            "bg_dilate_iter": int(args.bg_dilate_iter),
            "min_instance_area": int(args.min_instance_area),
        },
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
