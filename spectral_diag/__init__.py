"""Deterministic spectral colour-management diagnostics.

The public API is intentionally small.  The command line is the supported
entry point for repeatable runs; the functions exported here are useful for
embedding the same deterministic pipeline in tests or another application.
"""

from .errors import DiagnosticError, ErrorCode

__all__ = ["DiagnosticError", "ErrorCode"]
__version__ = "0.1.0"
