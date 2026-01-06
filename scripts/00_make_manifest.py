from pathlib import Path
import argparse

import _path  # noqa: F401; adds project root to sys.path for src imports
from src.dataset import discover_samples, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/U_NET/train"),
        help="Root folder of the extracted dataset.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"), help="Output manifest CSV path.")
    args = parser.parse_args()

    samples = discover_samples(args.data_root)
    write_manifest(samples, args.out)
    print(f"Wrote manifest with {len(samples)} samples to: {args.out}")


if __name__ == "__main__":
    main()
