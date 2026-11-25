"""Deterministic, fixed-point layered texture and material renderer."""

from .errors import BizError, EXIT_OK, EXIT_VALIDATION, EXIT_RENDER, EXIT_STATE

__all__ = [
    "BizError",
    "EXIT_OK",
    "EXIT_VALIDATION",
    "EXIT_RENDER",
    "EXIT_STATE",
]

__version__ = "1.0.0"
