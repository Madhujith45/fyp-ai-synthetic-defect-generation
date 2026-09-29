"""Tests for the generation pipeline and backends."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generation.backends import MockBackend, create_backend, BACKEND_REGISTRY
from src.generation.config import GenerationConfig


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    def test_defaults(self):
        config = GenerationConfig()
        assert config.backend == "mock"
        assert config.strength == 0.4
        assert config.image_size == (512, 512)

    def test_yaml_roundtrip(self, tmp_path: Path):
        config = GenerationConfig(
            prompt="test prompt",
            seed=123,
            strength=0.5,
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        loaded = GenerationConfig.from_yaml(yaml_path)
        assert loaded.prompt == "test prompt"
        assert loaded.seed == 123
        assert loaded.strength == 0.5


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_mock_registered(self):
        assert "mock" in BACKEND_REGISTRY

    def test_stable_diffusion_registered(self):
        assert "stable_diffusion" in BACKEND_REGISTRY

    def test_create_mock(self):
        backend = create_backend("mock")
        assert isinstance(backend, MockBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("nonexistent_backend")


class TestMockBackend:
    def test_generate_returns_results(self):
        from PIL import Image

        config = GenerationConfig(seed=42, num_images=2)
        backend = MockBackend()
        backend.load(config)

        # Create a simple test image
        test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

        results = backend.generate(
            image=test_image,
            prompt="Add a crack",
            config=config,
        )

        assert len(results) == 2
        assert results[0].backend_name == "mock"
        assert results[0].prompt == "Add a crack"
        assert results[0].seed == 42
        assert results[1].seed == 43

    def test_generate_modifies_image(self):
        from PIL import Image
        import numpy as np

        config = GenerationConfig(seed=42, num_images=1)
        backend = MockBackend()
        backend.load(config)

        test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        results = backend.generate(
            image=test_image,
            prompt="Add a defect",
            config=config,
        )

        # The output should differ from the input (mock adds noise)
        original_array = np.array(test_image)
        result_array = np.array(results[0].image)
        assert not np.array_equal(original_array, result_array)


# ---------------------------------------------------------------------------
# Pipeline integration test (mock backend, no GPU needed)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_full_pipeline_mock(self, tmp_path: Path):
        """Test the full pipeline end-to-end with mock backend."""
        from PIL import Image
        from src.generation.pipeline import run_generation_pipeline

        # Create a fake input image
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_image_path = input_dir / "test_casting.png"
        Image.new("RGB", (512, 512), color=(200, 200, 200)).save(str(input_image_path))

        output_dir = tmp_path / "output"

        config = GenerationConfig(
            input_image_path=input_image_path,
            prompt="Add a crack on the surface",
            output_dir=output_dir,
            backend="mock",
            seed=42,
            num_images=1,
        )

        saved_paths = run_generation_pipeline(config)

        assert len(saved_paths) == 1
        assert saved_paths[0].exists()
        assert saved_paths[0].suffix == ".png"

        # Check metadata sidecar exists
        meta_path = saved_paths[0].with_suffix(".json")
        assert meta_path.exists()
