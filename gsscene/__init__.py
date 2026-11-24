"""Deterministic multi-resolution Gaussian scene representation pipeline."""

__version__ = "0.1.0"

from .config import SceneConfig
from .pipeline import ScenePipeline

__all__ = ["SceneConfig", "ScenePipeline", "__version__"]
