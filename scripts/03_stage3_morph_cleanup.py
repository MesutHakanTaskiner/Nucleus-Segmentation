from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import read_image_bgr, build_binary_and_instance_masks
from src.viz import overlay_mask_on_image, save_png
from src.metrics import iou_dice
from src.stage2_threshold import run_stage2_threshold
from src.stage3_morphology import run_stage3_morphology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage3"))
    parser.add_argument("--blur-ksize", type=int, default=5)

    parser.add_argument("--open-ksize", type=int, default=3)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=30)

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
    pred2 = s2.pred_mask01

    # Stage 3
    s3 = run_stage3_morphology(
        pred2,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        min_area=args.min_area,
    )
    pred3 = s3.filtered01

    # Metrics
    iou2, dice2 = iou_dice(pred2, gt_merged01)
    iou3, dice3 = iou_dice(pred3, gt_merged01)

    # Save artifacts
    save_png(out / "image.png", img)
    save_png(out / "gt_mask_merged.png", (gt_merged01 * 255).astype(np.uint8))

    save_png(out / "pred_stage2.png", (pred2 * 255).astype(np.uint8))
    save_png(out / "pred_stage3_opened.png", (s3.opened01 * 255).astype(np.uint8))
    save_png(out / "pred_stage3_closed.png", (s3.closed01 * 255).astype(np.uint8))
    save_png(out / "pred_stage3_filled.png", (s3.filled01 * 255).astype(np.uint8))
    save_png(out / "pred_stage3_filtered.png", (pred3 * 255).astype(np.uint8))

    save_png(out / "overlay_stage2.png", overlay_mask_on_image(img, pred2, alpha=0.45))
    save_png(out / "overlay_stage3.png", overlay_mask_on_image(img, pred3, alpha=0.45))
    save_png(out / "overlay_gt.png", overlay_mask_on_image(img, gt_merged01, alpha=0.45))

    report = {
        "image_id": image_id,
        "num_gt_instances": int(gt_inst.max()),
        "stage2": {
            "otsu_used_invert": bool(s2.used_invert),
            "otsu_threshold": float(s2.otsu_thresh),
            "pred_fg_fraction": float(np.count_nonzero(pred2) / pred2.size),
            "iou": float(iou2),
            "dice": float(dice2),
        },
        "stage3": {
            "open_ksize": int(args.open_ksize),
            "close_ksize": int(args.close_ksize),
            "min_area": int(args.min_area),
            "pred_fg_fraction": float(np.count_nonzero(pred3) / pred3.size),
            "iou": float(iou3),
            "dice": float(dice3),
        },
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
