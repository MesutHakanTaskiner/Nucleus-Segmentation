from pathlib import Path

from .process_image import process_image


def process_directory(input_dir: Path, results_dir: Path, limit: int | None = 1) -> Path:
    images = []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            images.append(path)
            if limit and len(images) >= limit:
                break
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")
    process_image(images[0], results_dir)
    return images[0]
