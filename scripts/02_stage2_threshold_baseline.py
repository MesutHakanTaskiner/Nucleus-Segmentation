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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage2"))
    parser.add_argument("--blur-ksize", type=int, default=5)
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
    pred01 = s2.pred_mask01

    # Metrics vs GT merged (binary)
    iou, dice = iou_dice(pred01, gt_merged01)

    # Save artifacts
    save_png(out / "image.png", img)
    save_png(out / "gray.png", s2.gray)
    save_png(out / "blurred.png", s2.blurred)
    save_png(out / "pred_mask.png", (pred01 * 255).astype(np.uint8))
    save_png(out / "gt_mask_merged.png", (gt_merged01 * 255).astype(np.uint8))

    overlay_pred = overlay_mask_on_image(img, pred01, alpha=0.45)
    overlay_gt = overlay_mask_on_image(img, gt_merged01, alpha=0.45)
    save_png(out / "overlay_pred.png", overlay_pred)
    save_png(out / "overlay_gt.png", overlay_gt)

    # Write a small report.json
    report = {
        "image_id": image_id,
        "image_path": str(image_path),
        "num_gt_instances": int(gt_inst.max()),
        "otsu_used_invert": bool(s2.used_invert),
        "otsu_threshold": float(s2.otsu_thresh),
        "pred_fg_fraction": float(np.count_nonzero(pred01) / pred01.size),
        "gt_fg_fraction": float(np.count_nonzero(gt_merged01) / gt_merged01.size),
        "iou": float(iou),
        "dice": float(dice),
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
