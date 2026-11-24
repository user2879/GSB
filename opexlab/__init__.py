"""Deterministic Softmax/LayerNorm format-interoperability lab."""

from .config import LabConfig
from .engine import ExecutionEngine

__all__ = ["ExecutionEngine", "LabConfig"]
__version__ = "1.0.0"
