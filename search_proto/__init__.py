"""Deterministic local search migration prototype."""

from .config import PrototypeConfig, load_config
from .engine import SearchEngine
from .loader import load_inputs
from .model import Document, QuerySpec, StopwordTable

__all__ = [
    "Document",
    "PrototypeConfig",
    "QuerySpec",
    "SearchEngine",
    "StopwordTable",
    "load_config",
    "load_inputs",
]

