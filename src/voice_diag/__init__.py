"""Local deterministic voice-diagnostics component.

The package deliberately has no network, device, model-download, or container
dependency.  Inputs are ordinary WAV/JSON/text files produced by a fixed seed.
"""

from .errors import DiagnosticError, Issue
from .receipts import (
    A1Layout,
    ReceiptDocument,
    ReceiptRecord,
    parse_a1,
    parse_a2,
    to_a1,
    to_a2,
)

__all__ = [
    "A1Layout",
    "DiagnosticError",
    "Issue",
    "ReceiptDocument",
    "ReceiptRecord",
    "parse_a1",
    "parse_a2",
    "to_a1",
    "to_a2",
]

__version__ = "0.1.0"
