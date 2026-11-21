"""Deterministic fixed-point multimodal feature fusion.

The public entry points are intentionally small; most callers should use
``run_pipeline`` or the ``python -m multimodal_fusion`` CLI.
"""

from .config import FusionConfig, load_config
from .pipeline import run_pipeline

__all__ = ["FusionConfig", "load_config", "run_pipeline"]
__version__ = "1.0.0"
