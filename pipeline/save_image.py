from pathlib import Path

import cv2

from .ensure_dir import ensure_dir


def save_image(path: Path, image) -> None:
    ensure_dir(path)
    cv2.imwrite(str(path), image)
