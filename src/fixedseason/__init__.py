"""Deterministic fixed-point seasonal fitting.

Public API is intentionally small; the CLI and test bench build on these
objects.
"""

from .engine import FitEngine, run_once
from .model import FitConfig, SeriesRecord
from .loader import load_config, load_series_jsonl

__all__ = [
    "FitConfig",
    "FitEngine",
    "SeriesRecord",
    "load_config",
    "load_series_jsonl",
    "run_once",
]
