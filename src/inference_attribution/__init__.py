"""Deterministic batch-attribution diagnostics for local inference workers."""

from .errors import DiagnosticError
from .spec import load_merge_rules, load_requests, load_workers

__all__ = [
    "DiagnosticError",
    "load_merge_rules",
    "load_requests",
    "load_workers",
]

__version__ = "0.1.0"
