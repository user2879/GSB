"""Offline feature pipeline cache-hit diagnostics (fcdiag).

The package is deliberately self contained: it only depends on the Python
standard library at runtime.  ``numpy`` is an optional convenience used by the
sample generator and tests.
"""

from .errors import FcdError, InputInvalidError
from .models import (
    Artifact,
    Config,
    LedgerEntry,
    Pipeline,
    ReceiptRecord,
    Sample,
    StageDef,
)

__all__ = [
    "FcdError",
    "InputInvalidError",
    "Artifact",
    "Config",
    "LedgerEntry",
    "Pipeline",
    "ReceiptRecord",
    "Sample",
    "StageDef",
]

__version__ = "1.0.0"
