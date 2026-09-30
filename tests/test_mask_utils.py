"""Unit tests for mask generation utilities and mask-aware metrics."""

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

from src.generation.mask_utils import (
    compute_mask_aware_change,
    create_elliptical_mask,
    get_mask_statistics,
)


class TestCreateEllipticalMask:
    def test_output_is_grayscale(self):
        """Mask must be mode 'L'."""
        mask = create_elliptical_mask()
        assert mask.mode == "L"

    def test_output_size(self):
        """Mask must match requested size."""
        mask = create_elliptical_mask(image_size=(256, 128))
        assert mask.size == (256, 128)

    def test_default_size_512(self):
        """Default mask must be 512x512."""
        mask = create_elliptical_mask()
        assert mask.size == (512, 512)

    def test_contains_both_values(self):
        """Mask must contain both 0 and non-zero values (not all-black or all-white)."""
        mask = create_elliptical_mask()
        arr = np.array(mask)
        assert np.any(arr > 0), "Mask is entirely black"
        assert np.any(arr == 0), "Mask is entirely white"

    def test_mask_is_deterministic(self):
        """Same parameters must produce identical masks."""
        m1 = create_elliptical_mask(center=(200, 200), semi_axes=(30, 20))
        m2 = create_elliptical_mask(center=(200, 200), semi_axes=(30, 20))
        assert np.array_equal(np.array(m1), np.array(m2))

    def test_mask_coverage_is_small(self):
        """Default mask must cover less than 5% of the image."""
        mask = create_elliptical_mask()
        stats = get_mask_statistics(mask)
        assert stats["mask_percentage"] < 5.0, (
            f"Default mask covers {stats['mask_percentage']:.2f}%, expected < 5%"
        )

    def test_invalid_image_size_raises(self):
        with pytest.raises(ValueError):
            create_elliptical_mask(image_size=(0, 512))

    def test_invalid_semi_axes_raises(self):
        with pytest.raises(ValueError):
            create_elliptical_mask(semi_axes=(0, 10))


class TestGetMaskStatistics:
    def test_all_black_mask(self):
        """All-black mask should have 0% coverage."""
        mask = Image.new("L", (100, 100), 0)
        stats = get_mask_statistics(mask)
        assert stats["masked_pixels"] == 0
        assert stats["mask_percentage"] == 0.0

    def test_all_white_mask(self):
        """All-white mask should have 100% coverage."""
        mask = Image.new("L", (100, 100), 255)
        stats = get_mask_statistics(mask)
        assert stats["masked_pixels"] == 10000
        assert stats["mask_percentage"] == 100.0

    def test_partial_mask(self):
        """A mask with known pixel counts."""
        mask = Image.new("L", (10, 10), 0)
        arr = np.array(mask)
        arr[:2, :] = 255  # 20 pixels out of 100
        mask = Image.fromarray(arr)
        stats = get_mask_statistics(mask)
        assert stats["masked_pixels"] == 20
        assert np.isclose(stats["mask_percentage"], 20.0)


class TestComputeMaskAwareChange:
    def test_identical_images(self):
        """Identical images must produce 0% change everywhere."""
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        mask = create_elliptical_mask(
            image_size=(64, 64), center=(32, 32), semi_axes=(10, 5)
        )
        result = compute_mask_aware_change(img, img, mask)
        assert result["inside_mask_changed_pct"] == 0.0
        assert result["outside_mask_changed_pct"] == 0.0

    def test_change_only_inside_mask(self):
        """When only masked pixels differ, outside must show 0% change."""
        orig = Image.new("RGB", (64, 64), (100, 100, 100))
        gen = orig.copy()
        mask = Image.new("L", (64, 64), 0)
        mask_arr = np.array(mask)
        mask_arr[30:35, 30:35] = 255  # 25 pixels
        mask = Image.fromarray(mask_arr)

        gen_arr = np.array(gen)
        gen_arr[30:35, 30:35] = [200, 200, 200]  # diff of 100 per channel
        gen = Image.fromarray(gen_arr)

        result = compute_mask_aware_change(orig, gen, mask, threshold=10.0)
        assert result["inside_mask_changed_pct"] == 100.0
        assert result["outside_mask_changed_pct"] == 0.0

    def test_change_only_outside_mask(self):
        """When only unmasked pixels differ, inside must show 0% change."""
        orig = Image.new("RGB", (64, 64), (100, 100, 100))
        gen = orig.copy()
        mask = Image.new("L", (64, 64), 0)
        mask_arr = np.array(mask)
        mask_arr[30:35, 30:35] = 255
        mask = Image.fromarray(mask_arr)

        gen_arr = np.array(gen)
        gen_arr[0:5, 0:5] = [200, 200, 200]  # outside mask
        gen = Image.fromarray(gen_arr)

        result = compute_mask_aware_change(orig, gen, mask, threshold=10.0)
        assert result["inside_mask_changed_pct"] == 0.0
        assert result["outside_mask_changed_pct"] > 0.0

    def test_shape_mismatch_raises(self):
        """Mismatched image dimensions must raise ValueError."""
        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (32, 32))
        mask = Image.new("L", (64, 64))
        with pytest.raises(ValueError, match="mismatch"):
            compute_mask_aware_change(img1, img2, mask)
