"""Configuration dataclass for the synthetic defect generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GenerationConfig:
    """All parameters needed to run the generation pipeline.

    Attributes:
        input_image_path: Path to the source image (e.g. a normal casting).
        prompt: Text description of the defect to synthesise.
        negative_prompt: Things the model should avoid generating.
        output_dir: Directory where generated images are saved.
        backend: Which generation backend to use ("mock", "stable_diffusion").
        model_id: HuggingFace model identifier for diffusion backends.
        device: Torch device string ("cpu", "cuda", "cuda:0", etc.).
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale (higher = more prompt adherence).
        strength: img2img strength (0.0 = identical to input, 1.0 = fully generated).
            Lower values preserve more of the original structure.
        seed: Random seed for reproducibility. ``None`` means random.
        num_images: Number of images to generate per run.
        image_size: Target (width, height) for the model input.
    """

    # --- I/O ---
    input_image_path: Path = Path("data/raw/casting_512x512/casting_512x512/ok_front/cast_ok_0_1018.jpeg")
    prompt: str = "Add a small realistic surface crack on the casting"
    negative_prompt: str = "blurry, cartoon, drawing, low quality, deformed"
    output_dir: Path = Path("outputs/generation")

    # --- Backend ---
    backend: str = "mock"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"

    # --- Inference ---
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.4
    seed: int | None = 42
    num_images: int = 1
    image_size: tuple[int, int] = (512, 512)

    # --- Extra (for future extensions) ---
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> GenerationConfig:
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Convert string paths to Path objects
        for key in ("input_image_path", "output_dir"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = Path(raw[key])

        # Convert image_size list to tuple
        if "image_size" in raw and isinstance(raw["image_size"], list):
            raw["image_size"] = tuple(raw["image_size"])

        return cls(**raw)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "input_image_path": str(self.input_image_path),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "output_dir": str(self.output_dir),
            "backend": self.backend,
            "model_id": self.model_id,
            "device": self.device,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "seed": self.seed,
            "num_images": self.num_images,
            "image_size": list(self.image_size),
            "extra": self.extra,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
