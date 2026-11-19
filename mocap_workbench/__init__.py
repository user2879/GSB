"""Deterministic motion-capture retargeting workbench."""

from .errors import BusinessError, ErrorCode
from .generator import GenerationConfig, Scene, generate_scene
from .pipeline import solve_scene

__all__ = [
    "BusinessError",
    "ErrorCode",
    "GenerationConfig",
    "Scene",
    "generate_scene",
    "solve_scene",
]

__version__ = "0.1.0"
