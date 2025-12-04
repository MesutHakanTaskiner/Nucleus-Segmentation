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
    process_image(image_path, args.results_dir)
    print("Done. Outputs written to results/")


if __name__ == "__main__":
    main()
