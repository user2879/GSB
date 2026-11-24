"""Deterministic columnar-page crash recovery diagnostics."""

from .errors import (
    BudgetExceededError,
    CommitMarkerConflictError,
    DiagnosticError,
    RangeOutOfBoundsError,
    ReplayMismatchError,
    SnapshotMismatchError,
    WalValidationError,
)
from .models import Budget, Bundle, ColumnSpec, PageSpec, WalEntry
from .service import DiagnosticService

__version__ = "0.1.0"

__all__ = [
    "Budget",
    "Bundle",
    "ColumnSpec",
    "DiagnosticError",
    "DiagnosticService",
    "CommitMarkerConflictError",
    "RangeOutOfBoundsError",
    "ReplayMismatchError",
    "SnapshotMismatchError",
    "WalValidationError",
    "BudgetExceededError",
    "PageSpec",
    "WalEntry",
    "__version__",
]
