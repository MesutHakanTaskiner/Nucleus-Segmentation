from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_paths: List[Path]


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _find_dsb_style_samples(data_root: Path) -> List[Sample]:
    """
    Detect DSB2018-style layout:
      <id>/images/<id>.png
      <id>/masks/*.png
    """
    samples: List[Sample] = []
    for id_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        images_dir = id_dir / "images"
        masks_dir = id_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            continue

        # Usually exactly one image in images/
        img_files = [p for p in images_dir.iterdir() if _is_image_file(p)]
        if len(img_files) == 0:
            continue
        image_path = img_files[0]

        mask_files = [p for p in masks_dir.iterdir() if _is_image_file(p)]
        if len(mask_files) == 0:
            continue

        samples.append(Sample(image_id=id_dir.name, image_path=image_path, mask_paths=sorted(mask_files)))
    return samples


def _find_flat_samples(data_root: Path) -> List[Sample]:
    """
    Detect flat layout variants, e.g.:
      images/*.png
      masks/*.png   (same stem as images)
    or:
      img/ and mask/ etc.
    """
    # Common folder name guesses
    img_dirs = [data_root / "images", data_root / "img", data_root / "Imgs", data_root]
    mask_dirs = [data_root / "masks", data_root / "mask", data_root / "Masks"]

    img_dir = next((d for d in img_dirs if d.exists() and d.is_dir()), None)
    mask_dir = next((d for d in mask_dirs if d.exists() and d.is_dir()), None)

    if img_dir is None or mask_dir is None:
        return []

    img_files = sorted([p for p in img_dir.iterdir() if _is_image_file(p)])
    if len(img_files) == 0:
        return []

    # Map stem -> mask
    mask_map = {p.stem: p for p in mask_dir.iterdir() if _is_image_file(p)}

    samples: List[Sample] = []
    for img in img_files:
        m = mask_map.get(img.stem)
        if m is None:
            continue
        samples.append(Sample(image_id=img.stem, image_path=img, mask_paths=[m]))
    return samples


def discover_samples(data_root: Path) -> List[Sample]:
    """
    Discover samples under data_root without assuming a single fixed layout.
    Priority:
    1) DSB-style (<id>/images, <id>/masks)
    2) Flat images/masks folders with matching stems
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    dsb = _find_dsb_style_samples(data_root)
    if len(dsb) > 0:
        return dsb

    flat = _find_flat_samples(data_root)
    if len(flat) > 0:
        return flat

    raise RuntimeError(
        "Could not detect a supported dataset layout under data_root. "
        "Expected either DSB-style <id>/images & <id>/masks, or flat images/ + masks/ with matching stems."
    )


def write_manifest(samples: List[Sample], manifest_path: Path) -> None:
    """
    Write a CSV manifest so the rest of the pipeline does not care about folder layout.
    mask_paths are stored as ';' separated paths.
    """
    rows = []
    for s in samples:
        rows.append(
            {
                "image_id": s.image_id,
                "image_path": str(s.image_path.as_posix()),
                "mask_paths": ";".join([str(p.as_posix()) for p in s.mask_paths]),
                "num_masks": len(s.mask_paths),
            }
        )
    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)


def read_image_bgr(path: Path) -> np.ndarray:
    """Read an image as BGR uint8 (OpenCV default)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def read_mask_binary(path: Path) -> np.ndarray:
    """
    Read a mask as binary uint8 {0,1}.
    Accepts grayscale or color masks; non-zero becomes 1.
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    m_bin = (m > 0).astype(np.uint8)
    return m_bin


def build_binary_and_instance_masks(mask_paths: List[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of per-instance binary masks into:
      - merged binary mask (0/1)
      - instance label mask (0..N), where each nucleus gets a unique id.

    Assumption (true for DSB2018 ground truth): masks do not overlap. :contentReference[oaicite:2]{index=2}
    If overlaps exist, later masks will overwrite earlier labels in overlap pixels.
    """
    if len(mask_paths) == 0:
        raise ValueError("mask_paths is empty")

    first = read_mask_binary(mask_paths[0])
    h, w = first.shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.int32)

    for i, mp in enumerate(mask_paths, start=1):
        m = read_mask_binary(mp)
        if m.shape[:2] != (h, w):
            raise ValueError(f"Mask shape mismatch for {mp}: {m.shape} vs {(h, w)}")
        merged = np.maximum(merged, m)
        inst[m > 0] = i

    return merged, inst
