"""Self-contained runner script for the Stable Diffusion CUDA POC in Google Colab.

Usage in Google Colab:
    1. Ensure GPU runtime is enabled (Runtime -> Change runtime type -> T4 GPU).
    2. Run:
       !python scripts/run_colab_poc.py

This script:
    - Verifies CUDA is active.
    - Loads the real pretrained diffusion model (runwayml/stable-diffusion-v1-5).
    - Takes one normal casting from ok_front/ as input.
    - Applies prompt: "Add a small realistic surface crack on the casting."
    - Uses strength=0.38 to preserve casting geometry and concentric contours.
    - Saves the generated defect image and sidecar JSON metadata to outputs/generation/.
    - Prints the complete generation report (model, device, backend, time, paths).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generation.config import GenerationConfig
from src.generation.pipeline import run_generation_pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("  AI Industrial Synthetic Defect Generation - Colab CUDA POC")
    print("=" * 70)

    # 1. Environment & CUDA check
    try:
        import torch
        print(f"[*] PyTorch version : {torch.__version__}")
        print(f"[*] CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"[*] GPU Device      : {device_name} ({vram_gb} GB VRAM)")
            device = "cuda"
        else:
            print("[!] WARNING: CUDA is not available. Falling back to CPU.")
            device = "cpu"
    except ImportError:
        print("[!] PyTorch is not installed. Please install: pip install torch diffusers transformers accelerate")
        sys.exit(1)

    # 2. Configuration for Real Diffusion POC
    input_path = Path("data/raw/casting_512x512/casting_512x512/ok_front/cast_ok_0_1018.jpeg")
    if not input_path.exists():
        # Fallback search if nested directory varies
        candidates = list(Path("data").rglob("cast_ok_0_1018.jpeg"))
        if candidates:
            input_path = candidates[0]
        else:
            # Pick any ok_front image
            ok_candidates = list(Path("data").rglob("cast_ok_*.jpeg"))
            if ok_candidates:
                input_path = ok_candidates[0]
            else:
                print(f"[!] Error: Could not locate test casting image at '{input_path}'.")
                sys.exit(1)

    print(f"[*] Input casting   : {input_path}")

    config = GenerationConfig(
        input_image_path=input_path,
        prompt="Add a small realistic surface crack on the casting.",
        negative_prompt="blurry, low quality, distorted geometry, warped shape, cartoon, drawing, deformed casting, extra holes, changing object shape",
        output_dir=Path("outputs/generation"),
        backend="stable_diffusion",
        model_id="runwayml/stable-diffusion-v1-5",
        device=device,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.38,  # Calibrated for industrial casting geometry preservation
        seed=42,
        num_images=1,
        image_size=(512, 512),
    )

    # 3. Execute Pipeline
    t_start = time.perf_counter()
    saved_paths = run_generation_pipeline(config)
    total_pipeline_time = round(time.perf_counter() - t_start, 2)

    # 4. Comprehensive Report
    print("\n" + "=" * 70)
    print("  REAL GENERATION POC REPORT")
    print("=" * 70)
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

        print(f"  Model Used       : {config.model_id}")
        print(f"  Backend          : {config.backend}")
        print(f"  Device           : {config.device}")
        print(f"  Input Image      : {config.input_image_path}")
        print(f"  Defect Prompt    : \"{config.prompt}\"")
        print(f"  Strength (img2img): {config.strength} (preserves geometry & edges)")
        print(f"  Steps / CFG      : {config.num_inference_steps} steps, guidance={config.guidance_scale}")
        print(f"  Seed             : {config.seed}")
        print(f"  Inference Time   : {gen_time}")
        print(f"  Total Run Time   : {total_pipeline_time}s")
        print(f"  Generated Image  : {p}")
        print(f"  Metadata JSON    : {json_path}")
    print("=" * 70)
    print("✓ POC execution finished successfully.\n")


if __name__ == "__main__":
    main()
