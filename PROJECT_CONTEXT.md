# FYP PROJECT CONTEXT

## 1. Project Title
Few-Shot Synthetic Defect Image Generation for Engine Die-Casting Quality Inspection using Generative AI

---

## 2. Project Objective
Industrial manufacturing inspection faces severe data scarcity for defect detection: nominal (non-defective) castings are abundantly available, whereas physical defect samples (cracks, porosity, misruns, pinholes) are rare, hazardous, and economically punitive to produce. 

The objective of this Final Year Project (FYP) is to develop and evaluate a generative AI pipeline capable of synthesizing realistic, controllable industrial surface defects onto nominal casting images. By augmenting limited real defect samples with high-fidelity synthetic defect images, the research investigates downstream defect detection performance improvements, assessing whether generative data augmentation can bridge few-shot limitations in automated quality inspection.

---

## 3. Base Paper
- **Reference Paper:** *AnomalyControl: Few-Shot Anomaly Generation by ControlNet Inpainting*
- **Publication:** IEEE Access, 2024
- **Relation to Our Project:**
  *AnomalyControl* introduces a framework for generating few-shot surface anomalies by leveraging ControlNet-guided inpainting conditioned on anomaly layout masks and reference defect characteristics. This directly informs our planned pipeline: transitioning from unconstrained global image-to-image perturbations toward spatially bounded, mask-conditioned, defect-specific synthesis on engine die-castings without corrupting nominal component geometry.

---

## 4. Current Research Direction
The intended end-to-end research methodology is structured into a multi-phase pipeline:
1. **Few Real Defect Samples:** Ingest a small, highly constrained set of real casting defects representing target failure modes.
2. **Nominal Casting Images:** Utilize abundant defect-free casting images as base anatomical substrates.
3. **Industrial Expert Constraints:** Incorporate domain-specific physical rules (spatial boundaries, defect morphology, texture constraints).
4. **Generative Defect Synthesis:** Employ controllable generative models (Stable Diffusion v1.5 with inpainting / ControlNet guidance) to insert localized defect patterns.
5. **Quality Filtering / Evaluation:** Quantitatively and qualitatively evaluate generated samples, employing feature-space filtering (e.g., DINOv2 / CLIP) and structural fidelity checks to filter out implausible artifacts.
6. **Synthetic Dataset Assembly:** Aggregate approved synthetic defect samples with real nominal and real defect samples.
7. **Downstream Defect Detection Evaluation:** Benchmark defect classification/detection models (e.g., YOLO / CNN baselines) trained with vs. without synthetic augmentation to evaluate quantitative performance delta.

### Scope Distinction: Planned vs. Completed Work
- **Completed Work:** 
  - Local GPU environment setup and PyTorch CUDA acceleration verification.
  - Stage 1: Single-image real Stable Diffusion v1.5 img2img proof-of-concept (POC).
  - Stage 2: Small batch POC (3 nominal inputs × 2 seeds = 6 generations) with metadata tracking.
  - Stage 3: Image-level quantitative baseline evaluation (MAE, MSE, PSNR, SSIM, changed-pixel percentage) and test suite.
- **Planned Work (Not Yet Implemented):**
  - Spatially constrained defect inpainting / ControlNet conditioning (Stage 4).
  - DINOv2 / CLIP feature filtering and distribution metrics (FID/KID).
  - Fine-tuning of diffusion backends.
  - Downstream defect detector training (YOLO/ResNet) and detection performance evaluation.

---

## 5. Development Environment
- **Operating System:** Windows 11
- **Python Version:** 3.11.9 (64-bit)
- **Virtual Environment Path:** `D:\FYP\.venv`
- **PyTorch Version:** `2.14.0+cu126`
- **Torchvision Version:** `0.29.0+cu126`
- **CUDA Version:** `12.6` (PyTorch CUDA driver/runtime confirmed: `torch.cuda.is_available() == True`)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM:** 6 GB Dedicated GDDR6 VRAM
- **Key Installed Libraries:**
  - `diffusers`: `0.40.0`
  - `transformers`: `4.57.6`
  - `accelerate`: `1.13.0`
  - `scikit-image`: `0.26.0`
  - `pillow`: `12.3.0`
  - `numpy`: `2.4.6`
  - `pandas`: `3.0.6`
  - `pytest`: `9.1.1`
- **Model Weights Cache:** Cached locally under `C:\Users\ASUS\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5`

---

## 6. Dataset
- **Public Casting Dataset:** Submersible pump casting dataset for visual quality inspection.
- **Directory Location:** `D:\FYP\data\raw\casting_512x512\casting_512x512\`
- **Classes / Folders:**
  - `ok_front/`: 519 nominal (defect-free) front-view casting images.
  - `def_front/`: 781 defective front-view casting images.
  - Total Verified Images: 1,300 images.
- **Image Resolution:** 512×512 pixels, 8-bit RGB JPEG format.
- **Dataset Rules & Integrity:**
  - `data/raw/` must remain strictly **read-only**. Never modify, delete, resize, or overwrite raw files.
  - **Confidential Data Notice:** Confidential industrial/mentor data has **NOT** been added to the project. All work is conducted solely with the public casting dataset.

---

## 7. Project Structure
```text
D:\FYP\
|-- configs/
|   `-- generation.yaml             # Baseline generation parameters (model, prompt, strength, steps)
|-- data/
|   |-- raw/                        # READ-ONLY raw public casting dataset (1,300 images)
|   |-- processed/                  # Preprocessed artifacts (reserved for future pipeline stages)
|   `-- synthetic/                  # Synthetic dataset storage (reserved for scaled generation)
|-- outputs/
|   |-- generation/
|   |   |-- single_poc/             # Stage 1 single-generation output and metadata
|   |   `-- poc_batch/              # Stage 2 batch outputs (6 PNGs, 6 JSON sidecars, CSV/JSON summaries)
|   `-- evaluation/
|       |-- baseline_metrics.csv    # Stage 3 quantitative baseline metrics (per-image table)
|       |-- baseline_metrics.json   # Stage 3 quantitative baseline metrics & summary statistics
|       `-- baseline_report.md      # Stage 3 comprehensive markdown report
|-- scripts/
|   |-- generate.py                 # CLI generation script (supports mock & stable_diffusion backends)
|   |-- run_colab_poc.py            # Headless execution script for batch POC runs
|   `-- evaluate_baseline.py        # Stage 3 baseline evaluation script
|-- src/
|   |-- generation/
|   |   |-- __init__.py
|   |   |-- backends.py             # Backend implementations: MockBackend and StableDiffusionBackend
|   |   |-- config.py               # GenerationConfig dataclass with YAML serialization
|   |   `-- pipeline.py             # Orchestration pipeline and metadata saving
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- metrics.py              # MAE, MSE, PSNR, SSIM, and changed-pixel percentage metrics
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   `-- pipeline.py             # Initial preprocessing pipeline stubs
|   |-- training/
|   |   |-- __init__.py
|   |   `-- pipeline.py             # Downstream training pipeline stubs
|   `-- utils/
|       |-- __init__.py
|       `-- io.py                   # Safe I/O utility functions
`-- tests/
    |-- test_generation.py          # 9 unit tests for generation config, registry, mock, and pipeline
    `-- test_evaluation.py          # 5 unit tests for evaluation metrics (MAE, MSE, PSNR, SSIM, change %)
```

---

## 8. Completed Stages

### Stage 1 — Real Stable Diffusion POC
- **Model:** `runwayml/stable-diffusion-v1-5`
- **Pipeline:** Latent Image-to-Image (`StableDiffusionImg2ImgPipeline`) via `StableDiffusionBackend`
- **Hardware Acceleration:** CUDA on NVIDIA RTX 4050 Laptop GPU (6 GB VRAM, fp16 precision)
- **Input Image:** `data/raw/casting_512x512/casting_512x512/ok_front/cast_ok_0_1018.jpeg` (512×512 RGB)
- **Parameters:** Seed: 42, Strength: 0.38, Steps: 50, Guidance Scale: 7.5
- **Prompt:** `"Add a small realistic surface crack on the casting."`
- **Output:** `outputs/generation/single_poc/cast_ok_0_1018__seed42__s0.38__n0__20260929_161243.png`
- **Metadata:** `outputs/generation/single_poc/cast_ok_0_1018__seed42__s0.38__n0__20260929_161243.json`
- **Execution Time:** ~5.299 seconds
- **Verification:** Successfully executed on GPU; verified `StableDiffusionBackend` active with real neural network inference.

### Stage 2 — Small POC Batch
- **Setup:** 3 nominal casting inputs × 2 random seeds (42 and 99) = 6 total generation runs.
- **Inputs Evaluated:**
  - `cast_ok_0_1018.jpeg`
  - `cast_ok_0_1879.jpeg`
  - `cast_ok_0_2695.jpeg`
- **Execution:** 6/6 runs succeeded with 0 failures.
- **Average Generation Time:** ~4.601 seconds per 512×512 image.
- **Outputs & Metadata:** Saved in `outputs/generation/poc_batch/` (6 PNG files, 6 sidecar JSON files, `poc_batch_summary.csv`, and `poc_batch_summary.json`).
- **Test Status After Stage 2:** 9 passed, 0 failed in `tests/test_generation.py`.

### Stage 3 — Quantitative Baseline Evaluation
- **Objective:** Evaluate image-level pixel similarity and perturbation magnitude comparing the 6 generated synthetic images directly against their original nominal inputs.
- **Metrics Implemented:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Measure (SSIM)
  - Changed-Pixel Percentage (pixels with mean absolute RGB difference $> 10$ intensity levels)
- **Verified Numerical Results (6 Generations):**

| Input Image | Generated Output | Seed | MAE | MSE | PSNR (dB) | SSIM | Changed Pixels (%) | Gen Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cast_ok_0_1018.jpeg` | `...seed42...` | 42 | 7.4955 | 134.5655 | 26.8415 | 0.8447 | 21.23% | 5.751 |
| `cast_ok_0_1018.jpeg` | `...seed99...` | 99 | 7.3970 | 109.5598 | 27.7343 | 0.8828 | 23.22% | 4.680 |
| `cast_ok_0_1879.jpeg` | `...seed42...` | 42 | 9.6539 | 209.5720 | 24.9175 | 0.7545 | 32.48% | 4.217 |
| `cast_ok_0_1879.jpeg` | `...seed99...` | 99 | 8.8370 | 149.6308 | 26.3806 | 0.7821 | 30.60% | 4.391 |
| `cast_ok_0_2695.jpeg` | `...seed42...` | 42 | 8.1089 | 143.4694 | 26.5632 | 0.8671 | 24.32% | 4.165 |
| `cast_ok_0_2695.jpeg` | `...seed99...` | 99 | 7.0564 | 104.9113 | 27.9226 | 0.8737 | 20.57% | 4.403 |

- **Verified Summary Statistics:**
  - **MAE Mean:** `8.0915` (std: 0.9906)
  - **MSE Mean:** `141.9515` (std: 37.7082)
  - **PSNR Mean:** `26.7266 dB` (std: 1.0843)
  - **SSIM Mean:** `0.8342` (std: 0.0533)
  - **Changed Pixels (>10 intensity):** `25.40%` (std: 4.98%)
  - **Average Generation Time:** `4.601 s` (std: 0.592)
- **PyTest Verification Status:**
  - **14 passed, 0 failed, 0 warnings** in 2.48 seconds (`tests/test_generation.py` + `tests/test_evaluation.py`).

---

## 9. Important Metric Interpretation & Constraints
1. **Low-Level Similarity Only:** These metrics measure image-level pixel differences and global structural preservation relative to nominal inputs.
2. **Not Defect Realism:** High SSIM or PSNR does **NOT** prove that the synthetic defect is physically realistic or metallurgically valid. An identical image with no defect would yield perfect scores (SSIM=1.0, PSNR=$\infty$).
3. **Changed-Pixel Percentage != Defect Area:** Global img2img diffusion at strength 0.38 modulates the entire latent representation, inducing subtle lighting, shadow, and surface texture shifts across non-defect regions. The ~25.40% changed area represents this global latent modification, not the localized boundary of the crack.
4. **No Ground-Truth Target Pair:** Because synthetic defects are introduced into nominal components, there is no paired ground-truth defective counterpart.
5. **No Subjective Ranking:** Numerical scores do not establish whether an image is "good" or "bad" for downstream training without downstream detector validation.

---

## 10. Current Status
- **Stage 1 (Real Stable Diffusion POC):** COMPLETE
- **Stage 2 (Small POC Batch):** COMPLETE
- **Stage 3 (Quantitative Baseline Evaluation):** COMPLETE
- **Stage 4 (Spatially Controlled Defect Generation):** NOT STARTED

---

## 11. Next Planned Stage (Stage 4)
Stage 4 is intended to investigate **spatially controlled defect generation** using an appropriate Stable Diffusion v1.5 inpainting or ControlNet-compatible architecture.

The preliminary Stage 4 experiment will:
- Utilize one nominal casting image from the public dataset.
- Apply a deterministic, synthetic binary mask targeting a specific localized region of the casting surface.
- Strictly preserve the raw dataset (`data/raw/` remains read-only).
- Use fixed random seeds for exact reproducibility.
- Compare original nominal image, binary mask, and generated output.
- Compute the standard Stage 3 metrics (MAE, MSE, PSNR, SSIM).
- Introduce **mask-aware changed-pixel statistics** (measuring inside-mask perturbation vs. outside-mask preservation).
- Strictly adhere to RTX 4050 6 GB VRAM constraints (use fp16, efficient attention).
- Avoid model fine-tuning and avoid downstream detector training at this preliminary stage.

*(Note: Stage 4 must NOT be implemented until explicitly requested).*

---

## 12. Rules for Future Development
1. **Never modify or delete `data/raw/`:** The public dataset must remain intact and read-only.
2. **Confidentiality:** Never use, paste, or commit confidential industrial/mentor images or proprietary data with external AI coding assistants.
3. **Environment Stability:** Do NOT reinstall PyTorch, modify CUDA drivers, or recreate the `.venv` virtual environment unless explicitly required.
4. **Preserve Completed Experiments:** Do not overwrite existing outputs in `outputs/generation/` or `outputs/evaluation/`.
5. **Scientific Rigor:** Never claim industrial validity or defect realism based solely on low-level metrics (MAE/MSE/PSNR/SSIM).
6. **Reproducibility:** Always fix seeds and save sidecar metadata JSONs for every generated sample.
7. **Automated Testing:** Run `pytest` and ensure 100% pass rate after any code changes.

---

## 13. Useful Commands

### Activate Virtual Environment
```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

### Run Full Test Suite
```powershell
D:\FYP\.venv\Scripts\pytest.exe -v
```

### Run Generation Pipeline
```powershell
# Run CLI generator with default config (or specify custom prompt/backend)
D:\FYP\.venv\Scripts\python.exe scripts/generate.py --config configs/generation.yaml
```

### Run Stage 3 Baseline Evaluation
```powershell
D:\FYP\.venv\Scripts\python.exe scripts/evaluate_baseline.py
```

---

## 14. Output Locations
- **Stage 1 Output:**
  - PNG: `D:\FYP\outputs\generation\single_poc\cast_ok_0_1018__seed42__s0.38__n0__20260929_161243.png`
  - Metadata: `D:\FYP\outputs\generation\single_poc\cast_ok_0_1018__seed42__s0.38__n0__20260929_161243.json`
- **Stage 2 Outputs:**
  - Directory: `D:\FYP\outputs\generation\poc_batch\`
  - Batch Summaries: `poc_batch_summary.csv` and `poc_batch_summary.json`
- **Stage 3 Outputs:**
  - CSV Table: `D:\FYP\outputs\evaluation\baseline_metrics.csv`
  - JSON Metrics & Statistics: `D:\FYP\outputs\evaluation\baseline_metrics.json`
  - Markdown Report: `D:\FYP\outputs\evaluation\baseline_report.md`

---

## 15. Git / Backup Status
- **Repository Status:** `D:\FYP` is currently a local directory not initialized as a git repository (`fatal: not a git repository`).
- **Version Control Policy:** No automated git commit or push actions were performed. Version control initialization and remote backups should be managed manually by the project owner.
