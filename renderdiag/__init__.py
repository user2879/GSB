"""Deterministic multimodal conditional rendering diagnostic package."""

from .config import Config, load_config
from .engine import RenderEngine
from .errors import DiagnosticError

__all__ = ["Config", "DiagnosticError", "RenderEngine", "load_config"]
__version__ = "1.0.0"
