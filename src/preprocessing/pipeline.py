"""Image loading and preprocessing pipeline for industrial defect datasets."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils.io import build_logger, ensure_directory, iter_image_files, load_image, save_image


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the image preprocessing pipeline."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    target_size: tuple[int, int] = (512, 512)
    pad_color: tuple[int, int, int] = (0, 0, 0)
    output_extension: str = ".png"
    log_path: Path = Path("outputs/preprocessing/skipped_files.log")


@dataclass(frozen=True)
class PreprocessingSummary:
    """Summary statistics describing a preprocessing run."""

    total_files: int
    processed_files: int
    skipped_files: int
    output_dir: Path


def validate_target_size(target_size: tuple[int, int]) -> None:
    """Validate that a target image size is well-formed and positive."""
    if len(target_size) != 2:
        raise ValueError("target_size must contain exactly two integers.")

    if any(dimension <= 0 for dimension in target_size):
        raise ValueError("target_size dimensions must be positive integers.")


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize an image to fit within a target canvas while preserving aspect ratio."""
    validate_target_size(target_size)

    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full(
        (target_height, target_width, 3),
        pad_color,
        dtype=resized_image.dtype,
    )

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized_image

    return canvas


def is_valid_image(image_path: Path) -> bool:
    """Return ``True`` when an image can be successfully decoded by OpenCV."""
    return load_image(image_path) is not None


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path, output_extension: str) -> Path:
    """Build an output path that preserves the relative directory structure."""
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(output_extension)


def process_single_image(
    image_path: Path,
    config: PreprocessingConfig,
    logger: logging.Logger,
) -> bool:
    """Validate, resize, and save a single image file."""
    image = load_image(image_path)
    if image is None:
        logger.warning("Skipped corrupted or unreadable file: %s", image_path)
        return False

    processed_image = resize_with_aspect_ratio(
        image=image,
        target_size=config.target_size,
        pad_color=config.pad_color,
    )

    output_path = build_output_path(
        input_path=image_path,
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        output_extension=config.output_extension,
    )
    save_image(output_path, processed_image)
    return True


def run_preprocessing_pipeline(config: PreprocessingConfig) -> PreprocessingSummary:
    """Run the full preprocessing pipeline on all supported images in ``input_dir``."""
    validate_target_size(config.target_size)
    ensure_directory(config.output_dir)

    logger = build_logger("preprocessing_pipeline", config.log_path)
    image_paths = list(iter_image_files(config.input_dir))

    processed_files = 0
    skipped_files = 0

    for image_path in image_paths:
        if process_single_image(image_path=image_path, config=config, logger=logger):
            processed_files += 1
        else:
            skipped_files += 1

    return PreprocessingSummary(
        total_files=len(image_paths),
        processed_files=processed_files,
        skipped_files=skipped_files,
        output_dir=config.output_dir,
    )
