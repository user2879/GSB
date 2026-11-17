"""Deterministic local anomaly-diagnostic workbench."""

from .config import load_config_bundle
from .data import generate_dataset
from .pipeline import DetectionPipeline, run_full

__all__ = [
    "DetectionPipeline",
    "generate_dataset",
    "load_config_bundle",
    "run_full",
]

__version__ = "1.0.0"
