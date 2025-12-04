from pathlib import Path


def find_first_image(search_root: Path) -> Path:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for path in search_root.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            return path
    raise FileNotFoundError(f"No image files found under {search_root}")
