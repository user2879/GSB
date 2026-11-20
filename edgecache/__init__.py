"""Deterministic edge-cache consistency workbench.

The package intentionally depends only on the Python standard library at
runtime.  NumPy is used solely by the optional sample generator.
"""

from .errors import EdgeError
from .model import ContentRecord, ContentSnapshot, LoadedSpec, Policy

__all__ = [
    "ContentRecord",
    "ContentSnapshot",
    "EdgeError",
    "LoadedSpec",
    "Policy",
]

__version__ = "1.0.0"
