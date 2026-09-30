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
  - Stage 4: Spatially controlled defect generation via SD 1.5 inpainting with mask-aware evaluation.
  - Stage 4.1: Controlled ablation experiment (3 masks x 3 prompts x 2 seeds = 18 runs) evaluating mask size vs. defect clarity.
- **Planned Work (Not Yet Implemented):**
  - Multi-image / multi-mask inpainting batch experiments (Stage 5).
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

### Empirical Defect Dataset Analysis (Pre-Few-Shot Findings)
- **Defect Inventory (`def_front/`):** 781 images, all 512x512 JPEG, monochrome stored as 3-channel RGB (R=G=B). Mean intensity: 138.56 +/- 8.56 (vs. 148.95 +/- 9.06 for `ok_front/`).
- **Metadata Availability:** Zero annotations, bounding boxes, segmentation masks, or sub-category labels exist. Labels are strictly binary at the folder level. Filenames (`cast_def_0_<ID>.jpeg`) encode only sample IDs.
- **Defect Diversity:** `def_front/` aggregates multiple distinct physical casting failure modes rather than a single defect type:
  1. *Flash / Burrs / Finning:* Ragged excess material along outer mold parting line (highly prevalent).
  2. *Pinholes / Gas Blowholes / Porosity:* Small dark circular gas cavities on flat surfaces.
  3. *Surface Cracks / Scratches / Cuts:* Narrow linear fissures across the metallic face (relatively sparse).
  4. *Edge Chipping / Notches:* Discrete missing material on outer beveled flange or inner lip.
  5. *Shrinkage Cavities / Depressions:* Irregular dark voids formed during metal cooling.
- **Localization Feasibility:** Automated pixel subtraction against nominal images is unreliable due to minor rotation, translation (~5-15px), and optical zoom differences between samples. High-fidelity few-shot conditioning requires manual curation and annotation of reference defect exemplars.
- **Suitability for Few-Shot Research:** Suitable for few-shot exemplar conditioning (e.g., ControlNet inpainting conditioned on N=5 curated defect masks as in the *AnomalyControl* paper), but unsuitable for uncurated bulk fine-tuning without prior defect classification.

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
|   |   |-- poc_batch/              # Stage 2 batch outputs (6 PNGs, 6 JSON sidecars, CSV/JSON summaries)
|   |   |-- stage4_controlled/      # Stage 4 inpainting outputs (original, mask, generated, comparison, metadata, report)
|   |   `-- stage4_1_ablation/      # Stage 4.1 ablation outputs (18 runs, CSV/JSON summaries, comparison grids, report)
|   `-- evaluation/
|       |-- baseline_metrics.csv    # Stage 3 quantitative baseline metrics (per-image table)
|       |-- baseline_metrics.json   # Stage 3 quantitative baseline metrics & summary statistics
|       |-- baseline_report.md      # Stage 3 comprehensive markdown report
|       `-- defect_dataset_analysis/# Empirical defect analysis (report, dataset_summary.csv, representative_grid.png)
|-- scripts/
|   |-- generate.py                 # CLI generation script (supports mock & stable_diffusion backends)
|   |-- run_colab_poc.py            # Headless execution script for batch POC runs
|   |-- evaluate_baseline.py        # Stage 3 baseline evaluation script
|   |-- run_stage4_controlled.py    # Stage 4 controlled inpainting POC script
|   |-- run_stage4_1_ablation.py    # Stage 4.1 ablation experiment script (18 runs)
|   `-- analyze_defect_dataset.py   # Public defect dataset empirical analysis script
|-- src/
|   |-- generation/
|   |   |-- __init__.py
|   |   |-- backends.py             # Backend implementations: MockBackend and StableDiffusionBackend
|   |   |-- config.py               # GenerationConfig dataclass with YAML serialization
|   |   |-- mask_utils.py           # Deterministic mask generation and mask-aware change metrics
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
    |-- test_evaluation.py          # 5 unit tests for evaluation metrics (MAE, MSE, PSNR, SSIM, change %)
    `-- test_mask_utils.py          # 15 unit tests for mask creation and mask-aware change metrics
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

### Stage 4 — Spatially Controlled Defect Generation POC
- **Objective:** Compare spatially controlled inpainting (mask-guided) against the unconstrained img2img approach used in Stages 1–2. Confine pixel modifications to a specified mask region while preserving the rest of the casting image.
- **Model:** `runwayml/stable-diffusion-inpainting` (`StableDiffusionInpaintPipeline`)
  - Purpose-trained SD 1.5 inpainting variant with 9-channel UNet (4 latent + 4 masked-image latent + 1 mask channel).
  - VRAM usage: ~2.0 GB allocated in fp16 with attention slicing (well within 6 GB budget).
- **Input Image:** `data/raw/casting_512x512/casting_512x512/ok_front/cast_ok_0_1018.jpeg` (512×512 RGB)
- **Parameters:** Seed: 42, Steps: 50, Guidance Scale: 7.5
- **Prompt:** `"A small realistic surface crack on the metal casting"`
- **Mask Specification:**
  - Type: Deterministic elliptical mask
  - Center: (280, 260), Semi-axes: (25, 15) pixels, Rotation: 30°
  - Coverage: 0.4688% of image (1,229 / 262,144 pixels)
  - Bounding box: (257, 241, 302, 278)
  - **Note:** Synthetic POC mask. NOT a real industrial defect location.
- **Generation Time:** 10.769 seconds
- **Global Image-Level Metrics (Original vs. Generated):**

| Metric | Stage 4 Value | Stage 2 Mean (Baseline) |
|--------|:---:|:---:|
| MAE | 4.9311 | 8.0915 |
| MSE | 34.7619 | 141.9515 |
| PSNR (dB) | 32.7198 | 26.7266 |
| SSIM | 0.9576 | 0.8342 |
| Changed Pixels (>10 intensity) | 3.2848% | 25.40% |

- **Mask-Aware Change Statistics:**

| Region | Changed Pixels | Total Pixels | Changed % |
|--------|:---:|:---:|:---:|
| Inside Mask | 393 | 1,229 | 31.9772% |
| Outside Mask | 8,218 | 260,915 | 3.1497% |

- **Outputs:** `outputs/generation/stage4_controlled/` (original.png, mask.png, generated.png, comparison.png, metadata.json, report.md)
- **PyTest Verification Status:**
  - **29 passed, 0 failed, 0 warnings** in 10.11 seconds (`test_generation.py` + `test_evaluation.py` + `test_mask_utils.py`).

### Stage 4.1 — Controlled Inpainting Ablation Experiment
- **Objective:** Evaluate inpainting sensitivity across 3 deterministic mask sizes (small 15×10, medium 25×15, large 40×20), 3 defect-specific prompts, and 2 fixed seeds (42, 99) = 18 total generation runs on `cast_ok_0_1018.jpeg`.
- **Model:** `runwayml/stable-diffusion-inpainting` on RTX 4050 Laptop GPU (fp16).
- **Key Findings:** 18/18 runs completed successfully (~4.85 s/image). Spatial preservation remained excellent across all runs (outside-mask change 2.7%–3.8%, global SSIM > 0.95), but synthesized crack-like textures remained faint/subtle due to standard generic base weights, indicating the necessity of real defect conditioning/exemplars.
- **Outputs:** `outputs/generation/stage4_1_ablation/` (18 runs, 4 comparison grids, `ablation_summary.csv`, `report.md`).

### Stage 4.2 — Public Defect Dataset Analysis & Linear Defect Candidate Curation
- **Objective:** Empirically analyze the 781 images in `data/raw/.../def_front/`, screen for potential linear surface defects, and curate conservative candidate exemplars for few-shot conditioning research.
- **Screening & Inspection:**
  - Fast automated gradient/coherence screening across all 781 images (`scripts/screen_linear_defects.py`) generating `all_scores.csv` and multi-tier contact sheets.
  - Multi-stage visual inspection of candidates across zoom levels and edge-enhanced filters.
  - Identified that ~75-80% of defective images feature peripheral boundary flash/burrs, ~15-18% feature pinhole porosity, and only ~2-5% exhibit linear surface defects.
- **Curated Inventory:** 20 total candidates cataloged in `crack_candidates.csv` across three conservative categories:
  - 5 *candidate linear surface defects*
  - 8 *possible scratch/crack*
  - 7 *ambiguous*
- **Recommended Candidate Exemplars (Top 5):**
  1. `cast_def_0_5218.jpeg` (distinct crack on lower machined face)
  2. `cast_def_0_240.jpeg` (radial score/crack line across annular face)
  3. `cast_def_0_2263.jpeg` (transverse fracture across inner collar lip)
  4. `cast_def_0_976.jpeg` (transverse fracture cutting inner collar flange)
  5. `cast_def_0_3867.jpeg` (linear notch/crack propagating inward from outer bevel)
- **Outputs:** `outputs/evaluation/defect_dataset_analysis/` (`crack_candidates.csv`, `crack_candidate_grid.png`, `crack_candidate_report.md`).
- **Test Status:** 29 passed, 0 failed in `pytest`.

---

## 9. Important Metric Interpretation & Constraints
1. **Low-Level Similarity Only:** These metrics measure image-level pixel differences and global structural preservation relative to nominal inputs.
2. **Not Defect Realism:** High SSIM or PSNR does **NOT** prove that the synthetic defect is physically realistic or metallurgically valid. An identical image with no defect would yield perfect scores (SSIM=1.0, PSNR=$\infty$).
3. **Changed-Pixel Percentage != Defect Area:** Global img2img diffusion at strength 0.38 modulates the entire latent representation, inducing subtle lighting, shadow, and surface texture shifts across non-defect regions. The ~25.40% changed area represents this global latent modification, not the localized boundary of the crack.
4. **No Ground-Truth Target Pair:** Because synthetic defects are introduced into nominal components, there is no paired ground-truth defective counterpart.
5. **No Subjective Ranking:** Numerical scores do not establish whether an image is "good" or "bad" for downstream training without downstream detector validation.
6. **Mask-Aware Metrics Are Not Defect Localization:** Inside-mask changed percentage measures what fraction of the masked region shows visible pixel changes, NOT defect detection accuracy. Outside-mask changed percentage measures unintended perturbation leakage, NOT false positive rate.
7. **Curated Candidate Defect Labels:** Categories assigned during dataset curation (`candidate linear surface defect`, `possible scratch/crack`, `ambiguous`) are conservative empirical research categories, NOT ground-truth industrial quality labels.

---

## 10. Current Status
- **Stage 1 (Real Stable Diffusion POC):** COMPLETE
- **Stage 2 (Small POC Batch):** COMPLETE
- **Stage 3 (Quantitative Baseline Evaluation):** COMPLETE
- **Stage 4 (Spatially Controlled Defect Generation):** COMPLETE
- **Stage 4.1 (Controlled Inpainting Ablation):** COMPLETE
- **Stage 4.2 (Defect Dataset Analysis & Candidate Curation):** COMPLETE
- **Stage 5 (Multi-Image / Multi-Mask Inpainting Batch):** NOT STARTED

---

## 11. Next Planned Stage (Stage 5)
Stage 5 should investigate **scaling the controlled inpainting approach** across multiple images, masks, and seeds to establish statistical reliability.

Recommended Stage 5 experiments:
- Apply the inpainting pipeline to multiple nominal casting images (e.g., the same 3 used in Stage 2).
- Vary mask positions, sizes, and shapes across images.
- Use multiple seeds per image-mask pair.
- Compute aggregate mask-aware statistics (mean/std across the batch).
- Compare aggregate inpainting metrics against the Stage 2 unconstrained img2img baseline.
- Investigate different prompts targeting different defect types (crack, porosity, pinhole).
- Consider varying `guidance_scale` and `num_inference_steps` for quality sensitivity analysis.
- Evaluate whether lower outside-mask change can be achieved with prompt engineering or strength adjustment.

Future stages beyond Stage 5:
- DINOv2 / CLIP feature-space filtering and distribution metrics (FID/KID).
- Domain-specific fine-tuning of the inpainting model.
- Downstream defect detector training and evaluation.

*(Note: Stage 5 must NOT be implemented until explicitly requested).*

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

### Run Stage 4 Controlled Inpainting POC
```powershell
$env:HF_HUB_DISABLE_XET="1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"; D:\FYP\.venv\Scripts\python.exe scripts/run_stage4_controlled.py
```

### Run Stage 4.1 Inpainting Ablation Experiment
```powershell
D:\FYP\.venv\Scripts\python.exe scripts/run_stage4_1_ablation.py
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
- **Stage 4 Outputs:**
  - Directory: `D:\FYP\outputs\generation\stage4_controlled\`
  - Files: `original.png`, `mask.png`, `generated.png`, `comparison.png`, `metadata.json`, `report.md`
- **Stage 4.1 Outputs:**
  - Directory: `D:\FYP\outputs\generation\stage4_1_ablation\`
  - Subdirectories: `runs/` (18 individual run folders), `grids/` (`grid_all_18.png`, `grid_by_mask_size.png`, `grid_by_prompt.png`, `closeup_comparison.png`)
  - Summaries: `ablation_summary.csv`, `ablation_summary.json`, `report.md`
- **Defect Dataset Analysis Outputs:**
  - Directory: `D:\FYP\outputs\evaluation\defect_dataset_analysis\`
  - General Analysis: `report.md`, `dataset_summary.csv`, `representative_grid.png`
  - Screening Outputs: `screening/all_scores.csv`, `screening/screening_top30.png`, `screening/screening_top60.png`, `screening/face_closeup.png`
  - Stage 4.2 Curation Deliverables: `crack_candidates.csv`, `crack_candidate_grid.png`, `crack_candidate_report.md`
- **Model Weights Cache:**
  - SD 1.5 base: `C:\Users\ASUS\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5`
  - SD 1.5 inpainting: `C:\Users\ASUS\.cache\huggingface\hub\models--runwayml--stable-diffusion-inpainting`

---

## 15. Git / Backup Status
- **Repository:** `D:\FYP` is a git repository with remote `origin` at `https://github.com/Madhujith45/fyp-ai-synthetic-defect-generation.git`.
- **Branch:** `main`
- **Version Control Policy:** Stage 4 changes have not yet been committed. Version control operations should be managed manually by the project owner.
