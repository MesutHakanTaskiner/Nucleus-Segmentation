from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create parent directories for the given path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
