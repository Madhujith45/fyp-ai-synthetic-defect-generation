"""Generation package for synthetic defect data workflows.

Public API:
    - ``GenerationConfig`` — all pipeline parameters.
    - ``run_generation_pipeline`` — end-to-end orchestration.
    - ``create_backend`` — instantiate a backend by name.
    - ``GenerationResult`` — container for a generated image + metadata.
    - ``BACKEND_REGISTRY`` — registered backends (add new ones here).
"""

from src.generation.backends import (
    BACKEND_REGISTRY,
    GenerationBackend,
    GenerationResult,
    MockBackend,
    StableDiffusionBackend,
    create_backend,
)
from src.generation.config import GenerationConfig
from src.generation.pipeline import run_generation_pipeline

__all__ = [
    "BACKEND_REGISTRY",
    "GenerationBackend",
    "GenerationConfig",
    "GenerationResult",
    "MockBackend",
    "StableDiffusionBackend",
    "create_backend",
    "run_generation_pipeline",
]
