from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .compute_psnr_ssim_arrays import compute_psnr_ssim_arrays
from .ensure_dir import ensure_dir


def compare_psnr_ssim_to_reference(
    reference_image,
    targets: Iterable[Tuple[str, object]],
    output_csv: Path,
) -> None:
    """Compute PSNR/SSIM of targets against reference and write CSV."""
    rows = []
    for name, img in targets:
        psnr, ssim = compute_psnr_ssim_arrays(reference_image, img)
        rows.append({"name": name, "psnr": psnr, "ssim": ssim})
    df = pd.DataFrame(rows)
    ensure_dir(output_csv)
    df.to_csv(output_csv, index=False)
