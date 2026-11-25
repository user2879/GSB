"""Deterministic categorical cross-feature encoding diagnostics."""

from .config import LoadedBundle, load_bundle, load_bundle_from_dicts
from .engine import DiagnosticEngine
from .sample_data import TableData, generate_table

__all__ = [
    "DiagnosticEngine",
    "LoadedBundle",
    "TableData",
    "generate_table",
    "load_bundle",
    "load_bundle_from_dicts",
]

__version__ = "1.0.0"
