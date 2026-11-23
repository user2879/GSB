"""Deterministic near-duplicate detection over local Chinese short-text data."""

from .config import FingerprintConfig, load_config
from .engine import NearDuplicateEngine, SimulatedCrash
from .errors import BusinessError
from .text_store import TextRecord, load_text_jsonl

__all__ = [
    "BusinessError",
    "FingerprintConfig",
    "NearDuplicateEngine",
    "SimulatedCrash",
    "TextRecord",
    "load_config",
    "load_text_jsonl",
]

__version__ = "1.0.0"
