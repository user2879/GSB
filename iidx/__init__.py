"""Deterministic local inverted-index builder and compression scheduler."""

__version__ = "1.0.0"

from .config import AppConfig, ConfigLoadResult, load_config, DEFAULT_CONFIG
from .engine import Engine, RunResult
from .loader import load_inputs

__all__ = [
    "AppConfig",
    "ConfigLoadResult",
    "DEFAULT_CONFIG",
    "Engine",
    "RunResult",
    "load_config",
    "load_inputs",
]
