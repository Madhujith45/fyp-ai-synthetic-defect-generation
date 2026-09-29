"""CLI entry point for the synthetic defect generation pipeline.

Usage:
    # Run with defaults (mock backend, reads configs/generation.yaml):
    python scripts/generate.py

    # Override input image and prompt:
    python scripts/generate.py --input "data/raw/.../cast_ok_0_1082.jpeg" \\
                               --prompt "Add rust spots on the surface"

    # Run with Stable Diffusion on a CUDA machine:
    python scripts/generate.py --backend stable_diffusion --device cuda

    # Use a custom config file:
    python scripts/generate.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generation.config import GenerationConfig
from src.generation.pipeline import run_generation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic defect images from a normal casting + text prompt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation.yaml",
        help="Path to YAML config file (default: configs/generation.yaml).",
    )
    parser.add_argument("--input", type=str, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, help="Defect description prompt.")
    parser.add_argument("--negative-prompt", type=str, help="Negative prompt.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "stable_diffusion"],
        help="Generation backend to use.",
    )
    parser.add_argument("--device", type=str, help="Torch device (cpu, cuda, cuda:0).")
    parser.add_argument("--strength", type=float, help="img2img strength (0.0–1.0).")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--num-images", type=int, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    parser.add_argument("--steps", type=int, help="Number of inference steps.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Load base config from YAML
    config_path = Path(args.config)
    if config_path.exists():
        config = GenerationConfig.from_yaml(config_path)
        logging.info("Loaded config from: %s", config_path)
    else:
        config = GenerationConfig()
        logging.info("Using default config (no config file found at %s).", config_path)

    # Apply CLI overrides
    if args.input:
        config.input_image_path = Path(args.input)
    if args.prompt:
        config.prompt = args.prompt
    if args.negative_prompt:
        config.negative_prompt = args.negative_prompt
    if args.backend:
        config.backend = args.backend
    if args.device:
        config.device = args.device
    if args.strength is not None:
        config.strength = args.strength
    if args.seed is not None:
        config.seed = args.seed
    if args.num_images is not None:
        config.num_images = args.num_images
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.steps is not None:
        config.num_inference_steps = args.steps

    # Run
    saved_paths = run_generation_pipeline(config)

    print(f"\n{'=' * 60}")
    print(f"[OK] Generated {len(saved_paths)} image(s):")
    for p in saved_paths:
        json_path = p.with_suffix(".json")
        gen_time = "N/A"
        if json_path.exists():
            try:
                import json
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    gen_time = f"{meta.get('generation_time_sec', 'N/A')}s"
            except Exception:
                pass
        print(f"  -> Image Path     : {p}")
        print(f"     Model          : {config.model_id}")
        print(f"     Device         : {config.device}")
        print(f"     Backend        : {config.backend}")
        print(f"     Generation Time: {gen_time}")
        print(f"     Metadata Path  : {p.with_suffix('.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
