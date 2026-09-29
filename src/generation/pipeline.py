"""Main generation pipeline: load input → run backend → save output.

This module is the single orchestration point for synthetic defect generation.
It is backend-agnostic — the actual image generation is delegated to a
pluggable backend (see ``backends.py``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from PIL import Image

from src.generation.backends import GenerationResult, create_backend
from src.generation.config import GenerationConfig
from src.utils.io import ensure_directory

logger = logging.getLogger(__name__)


def load_input_image(config: GenerationConfig) -> Image.Image:
    """Load the source image specified in the config.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file cannot be decoded.
    """
    path = Path(config.input_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = Image.open(path).convert("RGB")
    logger.info("Loaded input image: %s (%dx%d)", path.name, image.width, image.height)
    return image


def build_output_filename(
    config: GenerationConfig,
    index: int,
    result: GenerationResult,
) -> str:
    """Build a descriptive filename for a generated image."""
    input_stem = Path(config.input_image_path).stem
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{input_stem}__seed{result.seed}__s{config.strength}__n{index}__{timestamp}.png"


def save_results(
    results: list[GenerationResult],
    config: GenerationConfig,
) -> list[Path]:
    """Save generated images and metadata to the output directory.

    Returns:
        List of paths to the saved images.
    """
    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)

    saved_paths: list[Path] = []

    for i, result in enumerate(results):
        filename = build_output_filename(config, i, result)
        image_path = output_dir / filename
        result.image.save(str(image_path))
        saved_paths.append(image_path)
        logger.info("Saved generated image: %s", image_path)

        # Save metadata sidecar
        gen_time = result.metadata.get("generation_time_sec") if result.metadata else None
        meta = {
            "source_image": str(config.input_image_path),
            "prompt": result.prompt,
            "negative_prompt": config.negative_prompt,
            "seed": result.seed,
            "backend": result.backend_name,
            "device": config.device,
            "model_id": config.model_id,
            "strength": config.strength,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "image_size": list(config.image_size),
            "generation_time_sec": gen_time,
            "output_file": str(image_path),
        }
        if result.metadata:
            meta["backend_metadata"] = result.metadata

        meta_path = image_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    return saved_paths


def run_generation_pipeline(config: GenerationConfig) -> list[Path]:
    """Execute the full generation pipeline end-to-end.

    Steps:
        1. Load the input image.
        2. Create and initialise the chosen backend.
        3. Generate synthetic image(s).
        4. Save outputs and metadata.
        5. Clean up backend resources.

    Args:
        config: Full generation configuration.

    Returns:
        List of paths to the saved generated images.
    """
    logger.info("=" * 60)
    logger.info("Starting generation pipeline")
    logger.info("  Backend      : %s", config.backend)
    logger.info("  Input        : %s", config.input_image_path)
    logger.info("  Prompt       : %s", config.prompt)
    logger.info("  Strength     : %s", config.strength)
    logger.info("  Seed         : %s", config.seed)
    logger.info("  Num images   : %d", config.num_images)
    logger.info("=" * 60)

    # 1. Load input
    input_image = load_input_image(config)

    # 2. Create backend
    backend = create_backend(config.backend)
    backend.load(config)

    try:
        # 3. Generate
        results = backend.generate(
            image=input_image,
            prompt=config.prompt,
            config=config,
        )

        # 4. Save
        saved_paths = save_results(results, config)

        logger.info(
            "Pipeline complete. %d image(s) saved to: %s",
            len(saved_paths),
            config.output_dir,
        )
        return saved_paths

    finally:
        # 5. Cleanup
        backend.unload()
