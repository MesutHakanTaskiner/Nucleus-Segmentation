import argparse
from pathlib import Path

from pipeline import find_first_image, process_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical nucleus segmentation pipeline (non-AI).")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input image. If omitted, the first image under data/ will be used.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (default: results)",
    )
    args = parser.parse_args()


    if args.input is None:
        image_path = find_first_image(Path("data"))
    else:
        image_path = args.input
    

    print(f"Processing image: {image_path}")
    results = process_image(image_path, args.results_dir)
    nucleus_count = results.get("nuclei_count")
    print("Done.")
    if nucleus_count is not None:
        print(f"Nucleus count: {nucleus_count}")
    if "metrics_csv" in results:
        print(f"IoU/Dice CSV: {results['metrics_csv']}")
    if "psnr_ssim_csv" in results:
        print(f"PSNR/SSIM CSV: {results['psnr_ssim_csv']}")
    if "features_csv" in results:
        print(f"Features CSV: {results['features_csv']}")
    if "final_mask" in results:
        print(f"Final mask: {results['final_mask']}")
    if "overlay_image" in results:
        print(f"Overlay image: {results['overlay_image']}")


if __name__ == "__main__":
    main()
