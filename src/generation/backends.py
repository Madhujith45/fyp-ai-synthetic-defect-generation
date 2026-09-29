"""Pluggable generation backends for the synthetic defect pipeline.

Each backend implements the same interface so the pipeline orchestrator can
swap backends transparently (e.g. local mock → local SD → cloud API).
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from src.generation.config import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for a single generated image and its metadata."""

    image: Image.Image
    prompt: str
    seed: int
    backend_name: str
    metadata: dict | None = None


class GenerationBackend(abc.ABC):
    """Abstract interface that all generation backends must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""

    @abc.abstractmethod
    def load(self, config: GenerationConfig) -> None:
        """Load / initialise the model or service connection.

        Called once before generate(). Backends should be lazy — do not
        download or allocate until this method is called.
        """

    @abc.abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        """Generate one or more synthetic images.

        Args:
            image: The source / input image (e.g. a normal casting).
            prompt: Text description of the modification to apply.
            config: Full generation configuration.

        Returns:
            A list of ``GenerationResult`` objects.
        """

    def unload(self) -> None:
        """Free resources. Override if the backend holds GPU memory etc."""


# ---------------------------------------------------------------------------
# Mock backend — runs anywhere, used for pipeline testing
# ---------------------------------------------------------------------------


class MockBackend(GenerationBackend):
    """Simulates generation by applying a visible synthetic overlay.

    This backend does NOT use any ML model. It applies a simple visual
    modification (noise + tint) so you can verify the full I/O pipeline
    works end-to-end without needing a GPU or downloading model weights.
    """

    @property
    def name(self) -> str:
        return "mock"

    def load(self, config: GenerationConfig) -> None:
        logger.info("MockBackend loaded (no model required).")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        rng = np.random.default_rng(config.seed)
        results: list[GenerationResult] = []

        for i in range(config.num_images):
            seed = (config.seed or 0) + i
            img_array = np.array(image).astype(np.float32)

            # Simulate a "defect" by adding localised noise + slight darkening
            h, w = img_array.shape[:2]
            # Random rectangular patch (roughly upper-left quadrant by default)
            y0 = rng.integers(0, h // 2)
            x0 = rng.integers(0, w // 2)
            patch_h = rng.integers(h // 8, h // 4)
            patch_w = rng.integers(w // 8, w // 4)
            y1 = min(y0 + patch_h, h)
            x1 = min(x0 + patch_w, w)

            t0 = time.perf_counter()
            # Add noise in patch
            noise = rng.normal(0, 30, img_array[y0:y1, x0:x1].shape)
            img_array[y0:y1, x0:x1] = np.clip(
                img_array[y0:y1, x0:x1] + noise, 0, 255
            )
            # Slight darkening in patch to simulate a crack/defect region
            img_array[y0:y1, x0:x1] *= 0.85

            gen_time = round(time.perf_counter() - t0, 4)
            out_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            results.append(
                GenerationResult(
                    image=out_image,
                    prompt=prompt,
                    seed=seed,
                    backend_name=self.name,
                    metadata={
                        "patch_region": (x0, y0, x1, y1),
                        "generation_time_sec": gen_time,
                    },
                )
            )

        logger.info(
            "MockBackend generated %d image(s) for prompt: '%s'",
            len(results),
            prompt,
        )
        return results


# ---------------------------------------------------------------------------
# Stable Diffusion img2img backend — requires CUDA GPU + diffusers
# ---------------------------------------------------------------------------


class StableDiffusionBackend(GenerationBackend):
    """Image-to-image generation using Stable Diffusion via HuggingFace diffusers.

    Designed for use on a CUDA-enabled machine. Will raise a clear error
    if loaded without GPU / required packages.
    """

    def __init__(self) -> None:
        self._pipe = None

    @property
    def name(self) -> str:
        return "stable_diffusion"

    def load(self, config: GenerationConfig) -> None:
        # Lazy imports — these packages may not be installed on every machine
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "The stable_diffusion backend requires 'torch' and 'diffusers'. "
                "Install them with: pip install torch diffusers transformers accelerate"
            ) from exc

        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Config requests device='cuda' but no CUDA GPU is available. "
                "Set device='cpu' (very slow) or run on a CUDA-enabled machine."
            )

        logger.info(
            "Loading Stable Diffusion model '%s' on device '%s'...",
            config.model_id,
            device,
        )

        dtype = torch.float16 if "cuda" in device else torch.float32
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self._pipe = self._pipe.to(device)

        # Memory optimisations
        if "cuda" in device:
            try:
                self._pipe.enable_attention_slicing()
            except Exception:
                pass

        logger.info("Stable Diffusion model loaded successfully.")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        import torch

        if self._pipe is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Resize input to model's expected size
        image = image.resize(config.image_size, Image.LANCZOS)

        generator = torch.Generator(device=config.device)
        results: list[GenerationResult] = []

        for i in range(config.num_images):
            seed = (config.seed or 0) + i
            generator.manual_seed(seed)

            if "cuda" in config.device and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                output = self._pipe(
                    prompt=prompt,
                    negative_prompt=config.negative_prompt,
                    image=image,
                    strength=config.strength,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    generator=generator,
                )

            if "cuda" in config.device and torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time = round(time.perf_counter() - t_start, 3)

            results.append(
                GenerationResult(
                    image=output.images[0],
                    prompt=prompt,
                    seed=seed,
                    backend_name=self.name,
                    metadata={
                        "model_id": config.model_id,
                        "device": config.device,
                        "strength": config.strength,
                        "guidance_scale": config.guidance_scale,
                        "num_inference_steps": config.num_inference_steps,
                        "generation_time_sec": gen_time,
                    },
                )
            )

        logger.info(
            "StableDiffusionBackend generated %d image(s) for prompt: '%s'",
            len(results),
            prompt,
        )
        return results

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("StableDiffusionBackend unloaded.")


# ---------------------------------------------------------------------------
# Backend registry — add new backends here
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type[GenerationBackend]] = {
    "mock": MockBackend,
    "stable_diffusion": StableDiffusionBackend,
}


def create_backend(backend_name: str) -> GenerationBackend:
    """Instantiate a backend by name.

    Raises:
        ValueError: If the backend name is not registered.
    """
    if backend_name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {available}"
        )
    return BACKEND_REGISTRY[backend_name]()
