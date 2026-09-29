# AI Performance Improvement Using Synthetic Training Data

## Project Objective

This project is a research-grade Python scaffold for studying how realistic synthetic industrial defect images can improve defect detection performance when only a small number of real defect images are available.

The current repository intentionally focuses on project structure, reproducibility, and workflow clarity. No machine learning or image generation models are implemented yet.

## Folder Structure

```text
.
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- configs/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- synthetic/
|-- docs/
|-- experiments/
|-- notebooks/
|-- outputs/
|-- src/
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   `-- pipeline.py
|   |-- generation/
|   |   |-- __init__.py
|   |   `-- pipeline.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- metrics.py
|   |-- training/
|   |   |-- __init__.py
|   |   `-- pipeline.py
|   `-- utils/
|       |-- __init__.py
|       `-- io.py
`-- tests/
```

## Setup Instructions

1. Create and activate a virtual environment.
2. Install project dependencies from `requirements.txt`.
3. Place real industrial defect images inside `data/raw/`.
4. Extend the pipelines and utilities as research requirements become clearer.

Example setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Planned Workflow

1. Collect and organize a small real defect dataset in `data/raw/`.
2. Build preprocessing steps for cleaning, resizing, annotation handling, and dataset preparation.
3. Add a synthetic data generation pipeline in `src/generation/`.
4. Evaluate synthetic data quality and downstream usefulness in `src/evaluation/`.
5. Integrate experiments and training orchestration in `src/training/` once a modeling phase begins.
6. Store generated artifacts, logs, and reports in `outputs/` and `docs/`.

## Design Principles

- Keep research code modular and easy to extend.
- Separate raw, processed, and synthetic datasets clearly.
- Support reproducible experiments with configurable pipelines.
- Delay model-specific decisions until the data workflow is stable.
