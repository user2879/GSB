"""Deterministic multimodal batch-inference ledger.

The package is intentionally dependency-light: NumPy is used only by the
worker/reference path, while the ledger itself uses integer arithmetic and the
Python standard library.
"""

from .version import __version__

__all__ = ["__version__"]
