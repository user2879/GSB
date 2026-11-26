"""Deterministic fixed-point multimodal conditional render pipeline."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import RenderConfig
from .codes import BizCode
from .fixedpoint import FpSpec
from .pipeline import PipelineState, RenderEngine, render_all, render_incremental

__all__ = [
    "__version__",
    "RenderConfig",
    "BizCode",
    "FpSpec",
    "PipelineState",
    "RenderEngine",
    "render_all",
    "render_incremental",
]
