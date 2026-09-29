"""Unit tests for the quantitative evaluation metrics module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_changed_pixel_percentage,
    evaluate_image_pair,
)


class TestEvaluationMetrics:
    def test_identical_images(self):
        """Identical images must yield MAE=0, MSE=0, SSIM=1.0, ChangedPct=0."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        res = evaluate_image_pair(img, img)

        assert res["mae"] == 0.0
        assert res["mse"] == 0.0
        assert res["ssim"] == 1.0
        assert res["changed_pixel_pct"] == 0.0
        assert np.isinf(res["psnr"]) or res["psnr"] > 99.0

    def test_constant_offset_mae_mse(self):
        """A uniform offset should produce mathematically exact MAE and MSE."""
        arr1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        arr2 = np.full((32, 32, 3), 110, dtype=np.uint8)

        mae = compute_mae(arr1, arr2)
        mse = compute_mse(arr1, arr2)

        assert np.isclose(mae, 10.0)
        assert np.isclose(mse, 100.0)

    def test_changed_pixel_percentage_thresholding(self):
        """Test that changed pixels are correctly identified based on threshold."""
        arr1 = np.zeros((10, 10, 3), dtype=np.uint8)
        arr2 = np.zeros((10, 10, 3), dtype=np.uint8)

        # 20 out of 100 pixels have RGB diff of 15 (exceeds threshold 10)
        arr2[:2, :, :] = 15
        # 30 out of 100 pixels have RGB diff of 5 (below threshold 10)
        arr2[2:5, :, :] = 5

        pct = compute_changed_pixel_percentage(arr1, arr2, threshold=10.0)
        assert np.isclose(pct, 20.0)

    def test_shape_mismatch_raises(self):
        """Images of differing dimensions must raise ValueError."""
        img1 = Image.new("RGB", (32, 32))
        img2 = Image.new("RGB", (64, 64))

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mae(img1, img2)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(img1, img2)

    def test_no_nan_or_inf_on_noisy_images(self):
        """Random images must produce finite, valid metrics."""
        np.random.seed(42)
        arr1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        res = evaluate_image_pair(arr1, arr2)
        for name, val in res.items():
            assert not np.isnan(val), f"Metric {name} is NaN"
            assert not np.isinf(val), f"Metric {name} is Inf"
