"""Deterministic normalization and adjudication for local batch receipts."""

from .model import MISSING, CanonicalReceipt, Failure, LedgerReport

__version__ = "1.0.0"
GLOBAL_LEDGER_VERSION = 1

__all__ = [
    "MISSING",
    "CanonicalReceipt",
    "Failure",
    "LedgerReport",
    "GLOBAL_LEDGER_VERSION",
]
