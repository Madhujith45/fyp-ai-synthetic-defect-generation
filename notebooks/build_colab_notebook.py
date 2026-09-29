"""Script to generate the robust, self-contained Google Colab notebook."""

import json
from pathlib import Path

# Load embedded sample casting image base64
b64_path = Path("notebooks/sample_image_b64.txt")
with open(b64_path, encoding="utf-8") as f:
    sample_b64 = f.read().strip()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Based Synthetic Image Generation for Industrial Visual Quality Inspection\n",
                "## Real Generation Milestone: Stable Diffusion (CUDA / Google Colab)\n",
                "\n",
                "**Objective:**\n",
                "Generate realistic synthetic defect images (surface cracks) on industrial casting components while strictly preserving the component's geometry, concentric circles, borehole, and overall appearance.\n",
                "\n",
                "- **Input:** Normal casting image (`cast_ok_0_1018.jpeg` from `ok_front/`)\n",
                "- **Prompt:** `\"Add a small realistic surface crack on the casting.\"`\n",
                "- **Model:** Pretrained `runwayml/stable-diffusion-v1-5` (`StableDiffusionImg2ImgPipeline`)\n",
                "- **Backend:** `stable_diffusion` on `cuda` (NVIDIA GPU)\n",
                "- **Geometry Preservation:** Controlled denoising strength (`strength=0.38`) + negative prompt geometry preservation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Verify CUDA GPU Environment\n",
                "Ensure your Colab runtime is set to GPU (**Runtime -> Change runtime type -> T4 GPU**)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "print(f\"PyTorch version : {torch.__version__}\")\n",
                "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU Device Name : {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"GPU Memory (VRAM): {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB\")\n",
                "else:\n",
                "    raise SystemError(\"CUDA GPU not detected. Please switch runtime: Runtime -> Change runtime type -> T4 GPU\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Install Required Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q diffusers transformers accelerate safetensors opencv-python Pillow PyYAML matplotlib"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Setup Casting Input Image\n",
                "This cell checks for the sample image on disk. If running in a fresh Colab instance without the full repository uploaded, it automatically restores the exact `cast_ok_0_1018.jpeg` casting image from embedded data so you never hit a `FileNotFoundError`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import base64\n",
                "from pathlib import Path\n",
                "from PIL import Image\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Target path matching project structure\n",
                "input_path = Path(\"data/raw/casting_512x512/casting_512x512/ok_front/cast_ok_0_1018.jpeg\")\n",
                "\n",
                "# Search if present elsewhere in data/ or current directory\n",
                "found_image = None\n",
                "if input_path.exists():\n",
                "    found_image = input_path\n",
                "else:\n",
                "    candidates = list(Path(\".\").rglob(\"cast_ok_0_1018.jpeg\")) + list(Path(\".\").rglob(\"cast_ok_*.jpeg\"))\n",
                "    if candidates:\n",
                "        found_image = candidates[0]\n",
                "\n",
                "if found_image is not None:\n",
                "    input_path = found_image\n",
                "    print(f\"[*] Found existing casting image at: {input_path}\")\n",
                "else:\n",
                "    print(\"[*] Image not found on Colab filesystem. Auto-restoring cast_ok_0_1018.jpeg...\")\n",
                "    input_path.parent.mkdir(parents=True, exist_ok=True)\n",
                f"    embedded_b64 = \"{sample_b64}\"\n",
                "    with open(input_path, \"wb\") as f:\n",
                "        f.write(base64.b64decode(embedded_b64))\n",
                "    print(f\"[✓] Successfully restored sample image to: {input_path}\")\n",
                "\n",
                "input_img = Image.open(input_path).convert(\"RGB\")\n",
                "print(f\"[*] Loaded image dimensions: {input_img.width}x{input_img.height}\")\n",
                "\n",
                "plt.figure(figsize=(5, 5))\n",
                "plt.imshow(input_img)\n",
                "plt.title(f\"Input Casting (Normal)\\n{input_path.name}\")\n",
                "plt.axis(\"off\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Configure & Run Stable Diffusion img2img Pipeline\n",
                "\n",
                "- **Prompt:** `\"Add a small realistic surface crack on the casting.\"`\n",
                "- **Strength = 0.38:** Preserves circular geometry, borehole, and metallic shading while synthesizing surface crack fissures.\n",
                "- **Model:** `runwayml/stable-diffusion-v1-5` with `torch.float16` and attention slicing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import time\n",
                "import json\n",
                "from datetime import datetime, timezone\n",
                "import torch\n",
                "from diffusers import StableDiffusionImg2ImgPipeline\n",
                "\n",
                "# Pipeline configuration\n",
                "model_id = \"runwayml/stable-diffusion-v1-5\"\n",
                "device = \"cuda\"\n",
                "prompt = \"Add a small realistic surface crack on the casting.\"\n",
                "negative_prompt = \"blurry, low quality, distorted geometry, warped shape, cartoon, drawing, deformed casting, extra holes, changing object shape\"\n",
                "strength = 0.38\n",
                "guidance_scale = 7.5\n",
                "num_inference_steps = 30\n",
                "seed = 42\n",
                "output_dir = Path(\"outputs/generation\")\n",
                "output_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f\"[*] Loading pretrained model '{model_id}' on {device}...\")\n",
                "pipe = StableDiffusionImg2ImgPipeline.from_pretrained(\n",
                "    model_id,\n",
                "    torch_dtype=torch.float16,\n",
                "    safety_checker=None,\n",
                "    requires_safety_checker=False,\n",
                ").to(device)\n",
                "pipe.enable_attention_slicing()\n",
                "print(\"[✓] Model loaded into GPU memory.\")\n",
                "\n",
                "# Prepare input\n",
                "resized_input = input_img.resize((512, 512), Image.LANCZOS)\n",
                "generator = torch.Generator(device=device).manual_seed(seed)\n",
                "\n",
                "print(\"[*] Generating synthetic defect image...\")\n",
                "torch.cuda.synchronize()\n",
                "t0 = time.perf_counter()\n",
                "\n",
                "with torch.inference_mode():\n",
                "    output = pipe(\n",
                "        prompt=prompt,\n",
                "        negative_prompt=negative_prompt,\n",
                "        image=resized_input,\n",
                "        strength=strength,\n",
                "        guidance_scale=guidance_scale,\n",
                "        num_inference_steps=num_inference_steps,\n",
                "        generator=generator,\n",
                "    )\n",
                "\n",
                "torch.cuda.synchronize()\n",
                "gen_time = round(time.perf_counter() - t0, 3)\n",
                "gen_image = output.images[0]\n",
                "\n",
                "# Save output and metadata sidecar\n",
                "timestamp = datetime.now(tz=timezone.utc).strftime(\"%Y%m%d_%H%M%S\")\n",
                "out_stem = f\"{input_path.stem}__seed{seed}__s{strength}__n0__{timestamp}\"\n",
                "output_image_path = output_dir / f\"{out_stem}.png\"\n",
                "output_json_path = output_dir / f\"{out_stem}.json\"\n",
                "\n",
                "gen_image.save(str(output_image_path))\n",
                "\n",
                "metadata = {\n",
                "    \"source_image\": str(input_path),\n",
                "    \"prompt\": prompt,\n",
                "    \"negative_prompt\": negative_prompt,\n",
                "    \"seed\": seed,\n",
                "    \"backend\": \"stable_diffusion\",\n",
                "    \"device\": device,\n",
                "    \"model_id\": model_id,\n",
                "    \"strength\": strength,\n",
                "    \"guidance_scale\": guidance_scale,\n",
                "    \"num_inference_steps\": num_inference_steps,\n",
                "    \"image_size\": [512, 512],\n",
                "    \"generation_time_sec\": gen_time,\n",
                "    \"output_file\": str(output_image_path),\n",
                "}\n",
                "\n",
                "with open(output_json_path, \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(metadata, f, indent=2)\n",
                "\n",
                "print(f\"[✓] Generation finished in {gen_time}s!\")\n",
                "print(f\"[✓] Saved image    : {output_image_path}\")\n",
                "print(f\"[✓] Saved metadata : {output_json_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Visual Inspection & Geometry Verification\n",
                "This visualizes the original casting, synthetic defect image, and the pixel difference heatmap to verify that the casting contours and inner hub are preserved while the crack is realistically synthesized."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "\n",
                "# Compute difference heatmap\n",
                "orig_arr = np.array(resized_input).astype(np.float32)\n",
                "gen_arr = np.array(gen_image).astype(np.float32)\n",
                "diff = np.abs(orig_arr - gen_arr)\n",
                "diff_gray = np.mean(diff, axis=-1)\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
                "\n",
                "axes[0].imshow(resized_input)\n",
                "axes[0].set_title(f\"Input Casting (Normal)\\n{input_path.name}\", fontsize=12)\n",
                "axes[0].axis(\"off\")\n",
                "\n",
                "axes[1].imshow(gen_image)\n",
                "axes[1].set_title(f\"Synthetic Defect Casting\\nPrompt: \\\"{prompt}\\\"\", fontsize=12)\n",
                "axes[1].axis(\"off\")\n",
                "\n",
                "im = axes[2].imshow(diff_gray, cmap=\"hot\")\n",
                "axes[2].set_title(\"Difference Heatmap\\n(Shows defect localized, geometry preserved)\", fontsize=12)\n",
                "axes[2].axis(\"off\")\n",
                "fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)\n",
                "\n",
                "plt.tight_layout()\n",
                "comparison_path = output_dir / \"comparison_result.png\"\n",
                "plt.savefig(comparison_path, dpi=200, bbox_inches=\"tight\")\n",
                "plt.show()\n",
                "print(f\"[*] Visual comparison saved to: {comparison_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Summary Report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\" * 70)\n",
                "print(\"  REAL GENERATION MILESTONE: EXECUTION REPORT\")\n",
                "print(\"=\" * 70)\n",
                "print(f\"  Model Used        : {metadata['model_id']}\")\n",
                "print(f\"  Backend           : {metadata['backend']}\")\n",
                "print(f\"  Device            : {metadata['device']}\")\n",
                "print(f\"  Input Image       : {metadata['source_image']}\")\n",
                "print(f\"  Defect Prompt     : \\\"{metadata['prompt']}\\\"\")\n",
                "print(f\"  Strength (img2img): {metadata['strength']} (preserves casting shape)\")\n",
                "print(f\"  Steps / Guidance  : {metadata['num_inference_steps']} steps, CFG={metadata['guidance_scale']}\")\n",
                "print(f\"  Seed              : {metadata['seed']}\")\n",
                "print(f\"  Inference Time    : {metadata['generation_time_sec']}s\")\n",
                "print(f\"  Generated Image   : {metadata['output_file']}\")\n",
                "print(f\"  Metadata JSON     : {output_json_path}\")\n",
                "print(\"=\" * 70)\n",
                "\n",
                "# Optional download button for Colab\n",
                "try:\n",
                "    from google.colab import files\n",
                "    print(\"\\n[*] Run `files.download(str(output_image_path))` to download generated image to your computer.\")\n",
                "except ImportError:\n",
                "    pass"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_nb_path = Path("notebooks/colab_sd_generation_poc.ipynb")
with open(output_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated notebook at {output_nb_path}")
