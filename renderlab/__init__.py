"""Deterministic local text-to-image and image-to-image diagnostics workbench."""

from .config import Failure, LoadResult, WorkbenchConfig, load_workbench
from .engine import RenderEngine, RenderReport
from .preview import PreviewBroker
from .strategy import StrategyDecision, select_strategy

__all__ = [
    "Failure",
    "LoadResult",
    "WorkbenchConfig",
    "load_workbench",
    "RenderEngine",
    "RenderReport",
    "PreviewBroker",
    "StrategyDecision",
    "select_strategy",
]

__version__ = "1.0.0"
