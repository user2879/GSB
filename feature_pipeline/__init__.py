"""Deterministic tabular feature-crossing and dimensionality pipeline."""

from .config import ENGINE_REVISION, load_pipeline_config
from .pipeline import run_pipeline

__all__ = ["ENGINE_REVISION", "load_pipeline_config", "run_pipeline"]
__version__ = "1.1.0"
