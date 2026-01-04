import argparse
from pathlib import Path
from pipeline import find_first_image, process_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical nucleus segmentation pipeline (non-AI).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Test_data/blood_cell_detection/BloodImage_00001.jpg"),
        help="Path to input image. If omitted, the first image under data/ will be used.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (default: results)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Test_data/blood_cell_detection/"),
        help="Process all images under this directory (each image gets its own subfolder in results-dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images when using --input-dir.",
    )
    args = parser.parse_args()

    def print_summary(image_path: Path, results: dict) -> None:
        nucleus_count = results.get("nuclei_count")
        print(f"Processed: {image_path}")
        if nucleus_count is not None:
            print(f"  Nucleus count: {nucleus_count}")
        if "metrics_csv" in results:
            print(f"  IoU/Dice CSV: {results['metrics_csv']}")
        if "psnr_ssim_csv" in results:
            print(f"  PSNR/SSIM CSV: {results['psnr_ssim_csv']}")
        if "features_csv" in results:
            print(f"  Features CSV: {results['features_csv']}")
        if "final_mask" in results:
            print(f"  Final mask: {results['final_mask']}")
        if "overlay_image" in results:
            print(f"  Overlay image: {results['overlay_image']}")

    tasks: list[tuple[Path, Path]] = []

    # Single input (explicit or auto-discovered)
    if args.input:
        single_out = args.results_dir / args.input.stem if args.input_dir else args.results_dir
        tasks.append((args.input, single_out))
    elif not args.input_dir:
        image_path = find_first_image(Path("data"))
        tasks.append((image_path, args.results_dir))

    # Directory inputs
    if args.input_dir:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        images = sorted([p for p in args.input_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()])
        if not images:
            raise SystemExit(f"No images found under {args.input_dir}")
        if args.limit:
            images = images[: args.limit]
        tasks.extend((img, args.results_dir / img.stem) for img in images)

    if not tasks:
        raise SystemExit("No inputs found to process.")

    print(f"Processing {len(tasks)} image(s). Base results dir: {args.results_dir}")
    for img_path, out_dir in tasks:
        print(f"- {img_path} -> {out_dir}")
        results = process_image(img_path, out_dir)
        print_summary(img_path, results)
    print("Done.")


if __name__ == "__main__":
    main()
