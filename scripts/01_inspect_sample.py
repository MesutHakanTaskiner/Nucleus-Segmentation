from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import read_image_bgr, build_binary_and_instance_masks
from src.viz import overlay_mask_on_image, save_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--idx", type=int, default=0, help="Row index in manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("results/inspect"))
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.idx < 0 or args.idx >= len(df):
        raise ValueError(f"idx out of range: {args.idx} (0..{len(df)-1})")

    row = df.iloc[args.idx]
    image_id = str(row["image_id"])
    image_path = Path(row["image_path"])
    mask_paths = [Path(p) for p in str(row["mask_paths"]).split(";") if p.strip()]

    img = read_image_bgr(image_path)
    merged01, inst = build_binary_and_instance_masks(mask_paths)

    # Stats
    nuclei_count = int(inst.max())
    fg_frac = float(np.count_nonzero(merged01) / merged01.size)

    # Save artifacts
    out = args.out_dir / image_id
    save_png(out / "image.png", img)
    save_png(out / "mask_merged.png", (merged01 * 255).astype(np.uint8))

    # Instance visualization: normalize labels to 0..255 just to see something
    inst_vis = (inst.astype(np.float32) / max(nuclei_count, 1) * 255.0).astype(np.uint8)
    save_png(out / "mask_instances_vis.png", inst_vis)

    overlay = overlay_mask_on_image(img, merged01, alpha=0.45)
    save_png(out / "overlay.png", overlay)

    print(f"image_id      : {image_id}")
    print(f"image_path    : {image_path}")
    print(f"num_masks     : {len(mask_paths)}")
    print(f"nuclei_count  : {nuclei_count}")
    print(f"fg_fraction   : {fg_frac:.4f}")
    print(f"saved to      : {out}")


if __name__ == "__main__":
    main()
