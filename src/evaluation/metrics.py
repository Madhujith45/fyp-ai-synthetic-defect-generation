"""Quantitative image-level similarity metrics module.

This module provides standard image-to-image similarity and change metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)
- Changed-Pixel Percentage (pixels exceeding mean absolute RGB change threshold)

IMPORTANT INTERPRETATION NOTICE:
These metrics compare generated synthetic images against their corresponding
original nominal images. They quantify image-level similarity and the magnitude
of low-level pixel perturbations introduced by the generation process.
They do NOT quantify "defect realism", biological or physical plausibility,
or downstream defect detection utility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_rgb_array(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Convert input image (PIL Image or numpy array) to uint8 RGB ndarray."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            # Grayscale to 3-channel
            arr = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA to RGB
            arr = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 3:
            arr = img
        else:
            raise ValueError(f"Unsupported image array shape: {img.shape}")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Expected PIL.Image.Image or np.ndarray, got {type(img)}")

    return arr


def compute_mae(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image],
) -> float:
    """Calculate Mean Absolute Error (MAE) across all pixel channels.

    Range: [0.0, 255.0]. Lower values indicate closer pixel intensity match.
    """
    arr1 = _to_rgb_array(img1).astype(np.float64)
    arr2 = _to_rgb_array(img2).astype(np.float64)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    val = float(np.mean(np.abs(arr1 - arr2)))
    if math.isnan(val) or math.isinf(val):
        raise ValueError("Computed MAE is NaN or Inf")
    return val


def compute_mse(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image],
) -> float:
    """Calculate Mean Squared Error (MSE) across all pixel channels.

    Range: [0.0, 65025.0]. Lower values indicate closer pixel intensity match.
    """
    arr1 = _to_rgb_array(img1).astype(np.float64)
    arr2 = _to_rgb_array(img2).astype(np.float64)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    val = float(np.mean((arr1 - arr2) ** 2))
    if math.isnan(val) or math.isinf(val):
        raise ValueError("Computed MSE is NaN or Inf")
    return val


def compute_psnr(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image],
    data_range: float = 255.0,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Higher values indicate greater fidelity to the original nominal image.
    If the two images are identical (MSE == 0), returns infinity.
    """
    arr1 = _to_rgb_array(img1)
    arr2 = _to_rgb_array(img2)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    if np.array_equal(arr1, arr2):
        return float("inf")

    val = float(peak_signal_noise_ratio(arr1, arr2, data_range=data_range))
    if math.isnan(val):
        raise ValueError("Computed PSNR is NaN")
    return val


def compute_ssim(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image],
    data_range: float = 255.0,
    channel_axis: int = 2,
) -> float:
    """Calculate Structural Similarity Index Measure (SSIM).

    Range: [-1.0, 1.0]. A value of 1.0 indicates identical structural content.
    Higher values mean the generated image preserves structural luminance,
    contrast, and pattern layout of the original image.
    """
    arr1 = _to_rgb_array(img1)
    arr2 = _to_rgb_array(img2)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    val = float(
        structural_similarity(
            arr1,
            arr2,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )
    if math.isnan(val) or math.isinf(val):
        raise ValueError("Computed SSIM is NaN or Inf")
    return val


def compute_changed_pixel_percentage(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image],
    threshold: float = 10.0,
) -> float:
    """Calculate percentage of pixels where mean absolute RGB difference exceeds threshold.

    For each pixel (r, c), the mean absolute RGB difference is:
        delta(r, c) = (|R1 - R2| + |G1 - G2| + |B1 - B2|) / 3.0

    A pixel is deemed 'changed' if delta(r, c) > threshold.

    Threshold documentation:
        - Default threshold = 10.0 (on a [0, 255] scale, ~3.9% intensity change).
        - Purpose: Filters out subtle sub-pixel rounding, compression noise,
          and imperceptible sensor artifacts.
        - Meaning: Quantifies the spatial footprint of discernible pixel changes
          introduced during generation.
        - Non-meaning: Does NOT indicate defect area, defect segmentation mask,
          or defect realism. Global background alterations also register as changed.

    Range: [0.0, 100.0] %.
    """
    arr1 = _to_rgb_array(img1).astype(np.float64)
    arr2 = _to_rgb_array(img2).astype(np.float64)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    per_pixel_mean_diff = np.mean(np.abs(arr1 - arr2), axis=2)
    changed_mask = per_pixel_mean_diff > threshold
    pct = float(np.mean(changed_mask) * 100.0)

    if math.isnan(pct) or math.isinf(pct):
        raise ValueError("Computed changed-pixel percentage is NaN or Inf")
    return pct


def evaluate_image_pair(
    original: Union[np.ndarray, Image.Image],
    generated: Union[np.ndarray, Image.Image],
    changed_threshold: float = 10.0,
) -> Dict[str, float]:
    """Compute all quantitative baseline evaluation metrics for a pair of images.

    Returns:
        dict with keys:
            - 'mae': Mean Absolute Error
            - 'mse': Mean Squared Error
            - 'psnr': Peak Signal-to-Noise Ratio (dB)
            - 'ssim': Structural Similarity Index Measure
            - 'changed_pixel_pct': Percentage of pixels exceeding difference threshold
    """
    return {
        "mae": compute_mae(original, generated),
        "mse": compute_mse(original, generated),
        "psnr": compute_psnr(original, generated),
        "ssim": compute_ssim(original, generated),
        "changed_pixel_pct": compute_changed_pixel_percentage(
            original, generated, threshold=changed_threshold
        ),
    }
