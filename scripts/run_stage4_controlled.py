"""Stage 4: Spatially Controlled Defect Generation POC.

This script runs a controlled inpainting experiment using Stable Diffusion 1.5
Inpainting model to generate a defect within a specified mask region while
preserving the rest of the casting image.

Usage:
    D:\\FYP\\.venv\\Scripts\\python.exe scripts/run_stage4_controlled.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import evaluate_image_pair
from src.generation.mask_utils import (
    compute_mask_aware_change,
    create_elliptical_mask,
    get_mask_statistics,
)


# ── Configuration ──────────────────────────────────────────────────────────

INPUT_IMAGE_PATH = _PROJECT_ROOT / "data" / "raw" / "casting_512x512" / "casting_512x512" / "ok_front" / "cast_ok_0_1018.jpeg"
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "generation" / "stage4_controlled"

# Inpainting model — purpose-trained SD 1.5 variant with 9-channel UNet
INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Generation parameters
SEED = 42
PROMPT = "A small realistic surface crack on the metal casting"
NEGATIVE_PROMPT = "blurry, cartoon, drawing, low quality, deformed, text, watermark"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = (512, 512)

# Mask parameters — small elliptical region on the casting surface
MASK_CENTER = (280, 260)
MASK_SEMI_AXES = (25, 15)
MASK_ROTATION_DEG = 30.0

# Changed-pixel threshold (same as Stage 3 for comparability)
CHANGE_THRESHOLD = 10.0


def run_stage4() -> None:
    """Execute the full Stage 4 controlled inpainting experiment."""

    # Lazy imports for GPU-dependent libraries
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    print("=" * 70)
    print("STAGE 4: Spatially Controlled Defect Generation POC")
    print("=" * 70)

    # ── Validate prerequisites ─────────────────────────────────────────
    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Stage 4 inpainting POC")

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    print(f"  GPU: {gpu_name} ({vram_gb} GB VRAM)")
    print(f"  Device: {device}")
    print(f"  Model: {INPAINT_MODEL_ID}")
    print(f"  Input: {INPUT_IMAGE_PATH.name}")
    print(f"  Seed: {SEED}")
    print()

    # ── Create output directory ────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare input image ───────────────────────────────────
    print("[1/7] Loading input image...")
    original = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    original.save(OUTPUT_DIR / "original.png")
    print(f"  Size: {original.size}, saved to original.png")

    # ── Create deterministic mask ──────────────────────────────────────
    print("[2/7] Creating deterministic mask...")
    mask = create_elliptical_mask(
        image_size=IMAGE_SIZE,
        center=MASK_CENTER,
        semi_axes=MASK_SEMI_AXES,
        rotation_deg=MASK_ROTATION_DEG,
    )
    mask.save(OUTPUT_DIR / "mask.png")
    mask_stats = get_mask_statistics(mask)
    print(f"  Mask coverage: {mask_stats['mask_percentage']:.4f}% of image")
    print(f"  Masked pixels: {mask_stats['masked_pixels']} / {mask_stats['total_pixels']}")
    print(f"  Bounding box: {mask_stats['mask_bbox']}")

    # ── Load inpainting model ──────────────────────────────────────────
    print("[3/7] Loading SD 1.5 inpainting model (fp16)...")
    t_load_start = time.perf_counter()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    t_load = round(time.perf_counter() - t_load_start, 3)
    print(f"  Model loaded in {t_load}s")

    # Report VRAM usage after loading
    vram_allocated = round(torch.cuda.memory_allocated() / 1024**3, 3)
    vram_reserved = round(torch.cuda.memory_reserved() / 1024**3, 3)
    print(f"  VRAM allocated: {vram_allocated} GB, reserved: {vram_reserved} GB")

    # ── Run inpainting generation ──────────────────────────────────────
    print("[4/7] Running inpainting generation...")
    generator = torch.Generator(device=device).manual_seed(SEED)

    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=original,
            mask_image=mask,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            height=IMAGE_SIZE[1],
            width=IMAGE_SIZE[0],
        )

    torch.cuda.synchronize()
    gen_time = round(time.perf_counter() - t_gen_start, 3)

    generated = result.images[0]
    generated.save(OUTPUT_DIR / "generated.png")
    print(f"  Generation time: {gen_time}s")
    print(f"  Output saved to generated.png")

    # ── Unload model and free VRAM ─────────────────────────────────────
    del pipe
    del result
    torch.cuda.empty_cache()

    # ── Compute Stage 3 metrics ────────────────────────────────────────
    print("[5/7] Computing image-level metrics...")
    global_metrics = evaluate_image_pair(original, generated, changed_threshold=CHANGE_THRESHOLD)
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Validate no NaN/Inf
    for k, v in global_metrics.items():
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Metric {k} produced invalid value: {v}")

    # ── Compute mask-aware change statistics ───────────────────────────
    print("[6/7] Computing mask-aware change statistics...")
    mask_metrics = compute_mask_aware_change(original, generated, mask, threshold=CHANGE_THRESHOLD)
    print(f"  Inside mask changed:  {mask_metrics['inside_mask_changed_pct']:.4f}%"
          f" ({mask_metrics['inside_mask_changed_pixels']}/{mask_metrics['inside_mask_total_pixels']} pixels)")
    print(f"  Outside mask changed: {mask_metrics['outside_mask_changed_pct']:.4f}%"
          f" ({mask_metrics['outside_mask_changed_pixels']}/{mask_metrics['outside_mask_total_pixels']} pixels)")

    # ── Create comparison image ────────────────────────────────────────
    print("[7/7] Creating visual comparison...")
    comparison = create_comparison_image(original, mask, generated)
    comparison.save(OUTPUT_DIR / "comparison.png")
    print("  Saved comparison.png")

    # ── Save metadata JSON ─────────────────────────────────────────────
    metadata = {
        "stage": "Stage 4 — Spatially Controlled Defect Generation POC",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "input_image": str(INPUT_IMAGE_PATH),
        "input_filename": INPUT_IMAGE_PATH.name,
        "model_id": INPAINT_MODEL_ID,
        "pipeline": "StableDiffusionInpaintPipeline",
        "device": device,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "seed": SEED,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "image_size": list(IMAGE_SIZE),
        "mask_params": {
            "type": "elliptical",
            "center": list(MASK_CENTER),
            "semi_axes": list(MASK_SEMI_AXES),
            "rotation_deg": MASK_ROTATION_DEG,
            "note": "Synthetic POC mask. NOT a real industrial defect location.",
        },
        "mask_statistics": mask_stats,
        "model_load_time_sec": t_load,
        "generation_time_sec": gen_time,
        "global_metrics": {k: round(v, 4) for k, v in global_metrics.items()},
        "mask_aware_metrics": mask_metrics,
        "change_threshold": CHANGE_THRESHOLD,
        "interpretation_notice": (
            "All metrics are image-level pixel-change measurements. "
            "They do NOT measure defect realism, industrial validity, "
            "or defect detection performance."
        ),
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("  Saved metadata.json")

    # ── Generate Markdown report ───────────────────────────────────────
    report = generate_report(metadata)
    with open(OUTPUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("  Saved report.md")

    print()
    print("=" * 70)
    print("Stage 4 completed successfully!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


def create_comparison_image(
    original: Image.Image,
    mask: Image.Image,
    generated: Image.Image,
) -> Image.Image:
    """Create a side-by-side comparison: Original | Mask | Generated."""
    w, h = original.size
    padding = 10
    label_height = 30
    total_w = w * 3 + padding * 4
    total_h = h + padding * 2 + label_height

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Paste images
    y_offset = label_height + padding
    canvas.paste(original, (padding, y_offset))
    # Convert mask to RGB for display (with red overlay)
    mask_display = original.copy()
    mask_arr = np.array(mask)
    orig_arr = np.array(mask_display)
    # Overlay mask region in red
    overlay = orig_arr.copy()
    overlay[mask_arr == 255] = [255, 50, 50]
    blended = (orig_arr * 0.5 + overlay * 0.5).astype(np.uint8)
    mask_display = Image.fromarray(blended)
    canvas.paste(mask_display, (padding * 2 + w, y_offset))
    canvas.paste(generated, (padding * 3 + w * 2, y_offset))

    # Labels
    labels = ["Original", "Mask Region", "Generated (Inpainted)"]
    for i, label in enumerate(labels):
        x = padding + i * (w + padding) + w // 2
        draw.text((x, 8), label, fill=(220, 220, 220), anchor="mt")

    return canvas


def generate_report(metadata: dict) -> str:
    """Generate a Stage 4 markdown report."""
    m = metadata
    gm = m["global_metrics"]
    mm = m["mask_aware_metrics"]
    ms = m["mask_statistics"]
    mp = m["mask_params"]

    lines = [
        "# Stage 4: Spatially Controlled Defect Generation — POC Report",
        "",
        "## 1. Objective",
        "",
        "Compare spatially controlled inpainting (mask-guided) against the unconstrained",
        "img2img approach used in Stages 1–2. The goal is to confine pixel modifications",
        "to a specified mask region while preserving the rest of the casting image.",
        "",
        "---",
        "",
        "## 2. Method",
        "",
        f"- **Model:** `{m['model_id']}` (`{m['pipeline']}`)",
        f"- **Device:** {m['device']} — {m['gpu']} ({m['vram_gb']} GB VRAM)",
        f"- **Input Image:** `{m['input_filename']}`",
        f"- **Seed:** {m['seed']}",
        f"- **Prompt:** \"{m['prompt']}\"",
        f"- **Negative Prompt:** \"{m['negative_prompt']}\"",
        f"- **Steps:** {m['num_inference_steps']}, **Guidance Scale:** {m['guidance_scale']}",
        f"- **Resolution:** {m['image_size'][0]}×{m['image_size'][1]}",
        "",
        "### Mask Specification",
        "",
        f"- **Type:** {mp['type']}",
        f"- **Center:** ({mp['center'][0]}, {mp['center'][1]})",
        f"- **Semi-axes:** ({mp['semi_axes'][0]}, {mp['semi_axes'][1]}) pixels",
        f"- **Rotation:** {mp['rotation_deg']}°",
        f"- **Mask Coverage:** {ms['mask_percentage']:.4f}% of image ({ms['masked_pixels']} pixels)",
        f"- **Bounding Box:** {ms['mask_bbox']}",
        f"- **Note:** {mp['note']}",
        "",
        "---",
        "",
        "## 3. Timing",
        "",
        f"- **Model Load Time:** {m['model_load_time_sec']}s",
        f"- **Generation Time:** {m['generation_time_sec']}s",
        "",
        "---",
        "",
        "## 4. Global Image-Level Metrics (Original vs. Generated)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MAE | {gm['mae']:.4f} |",
        f"| MSE | {gm['mse']:.4f} |",
        f"| PSNR (dB) | {gm['psnr']:.4f} |",
        f"| SSIM | {gm['ssim']:.4f} |",
        f"| Changed Pixels (>{m['change_threshold']} intensity) | {gm['changed_pixel_pct']:.4f}% |",
        "",
        "---",
        "",
        "## 5. Mask-Aware Change Statistics",
        "",
        "| Region | Changed Pixels | Total Pixels | Changed % |",
        "|--------|---------------|--------------|-----------|",
        f"| **Inside Mask** | {mm['inside_mask_changed_pixels']} | {mm['inside_mask_total_pixels']} | {mm['inside_mask_changed_pct']:.4f}% |",
        f"| **Outside Mask** | {mm['outside_mask_changed_pixels']} | {mm['outside_mask_total_pixels']} | {mm['outside_mask_changed_pct']:.4f}% |",
        "",
        "---",
        "",
        "## 6. Comparison with Stage 2 Baseline (Unconstrained img2img)",
        "",
        "| Metric | Stage 2 Mean (img2img) | Stage 4 (Inpainting) | Interpretation |",
        "|--------|----------------------|---------------------|----------------|",
        f"| MAE | 8.0915 | {gm['mae']:.4f} | Lower = less global change |",
        f"| PSNR (dB) | 26.7266 | {gm['psnr']:.4f} | Higher = better fidelity |",
        f"| SSIM | 0.8342 | {gm['ssim']:.4f} | Higher = better structure preservation |",
        f"| Changed Pixels % | 25.40% | {gm['changed_pixel_pct']:.4f}% | Lower = more spatially focused |",
        f"| Outside Mask Changed % | N/A (no mask) | {mm['outside_mask_changed_pct']:.4f}% | Lower = better spatial control |",
        "",
        "---",
        "",
        "## 7. Interpretation and Limitations",
        "",
        "1. **What these metrics measure:** Image-level pixel differences and the spatial",
        "   distribution of pixel changes relative to a synthetic mask boundary.",
        "",
        "2. **What they do NOT measure:**",
        "   - Defect realism or physical plausibility",
        "   - Industrial defect detection accuracy",
        "   - Whether generated pixels inside the mask represent a real crack",
        "   - Ground-truth defect segmentation quality",
        "",
        "3. **Mask is synthetic:** The elliptical mask was placed at a fixed position",
        "   for technical demonstration. It does not correspond to a real or",
        "   expert-identified defect location.",
        "",
        "4. **Single-image POC:** Results are from one image with one seed.",
        "   Statistical significance requires evaluation across multiple images and seeds.",
        "",
        "5. **No fine-tuning:** The model was used as-is from HuggingFace.",
        "   Domain-specific fine-tuning may improve defect texture quality.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_stage4()
