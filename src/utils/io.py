"""Reusable filesystem and image I/O utilities for research workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png")


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_image_files(directory: Path) -> Iterator[Path]:
    """Yield supported image files from a directory tree in sorted order."""
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield file_path


def load_image(image_path: Path) -> np.ndarray | None:
    """Load an image with OpenCV, returning ``None`` when decoding fails."""
    try:
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
    except OSError:
        return None

    if image_bytes.size == 0:
        return None

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    return image


def save_image(image_path: Path, image: np.ndarray) -> None:
    """Save an image to disk, creating parent directories as needed."""
    ensure_directory(image_path.parent)
    success = cv2.imwrite(str(image_path), image)
    if not success:
        raise ValueError(f"Failed to save image to '{image_path}'.")


def build_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    """Create a logger that writes to the console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        ensure_directory(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
