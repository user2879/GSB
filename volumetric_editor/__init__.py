"""Deterministic transparent-media and volumetric-light editor."""

from .errors import BusinessError, ErrorCode
from .scene import Scene, generate_scene, load_scene, save_scene

__all__ = [
    "BusinessError",
    "ErrorCode",
    "Scene",
    "generate_scene",
    "load_scene",
    "save_scene",
]

__version__ = "1.0.0"
