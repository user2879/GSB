"""Local object-storage sharded upload diagnostics workbench.

The package is intentionally dependency-light: NumPy is used only for fast,
deterministic byte generation; the ledger, scheduling, and diagnostics use the
Python standard library.
"""

from .errors import DiagnosticError, DigestError, LeaseError, SnapshotError, SpecError
from .models import LeasePolicy, ObjectSpec, SessionSpec, load_spec, load_spec_data

__all__ = [
    "DiagnosticError",
    "DigestError",
    "LeaseError",
    "SnapshotError",
    "SpecError",
    "LeasePolicy",
    "ObjectSpec",
    "SessionSpec",
    "load_spec",
    "load_spec_data",
]

__version__ = "1.0.0"
