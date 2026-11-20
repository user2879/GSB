"""Deterministic streaming instruction parser.

The package intentionally has no dependency beyond the Python standard library
and NumPy.  All decoding is performed on synthetic frame-level score tables.
"""

from .app import IncrementalApp
from .engine import full_decode, semantic_payload

__all__ = ["IncrementalApp", "full_decode", "semantic_payload"]
__version__ = "1.0.0"
