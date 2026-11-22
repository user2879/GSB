"""Local block snapshot and rollback diagnostics."""

__version__ = "1.0.0"

from .errors import DiagnosticError
from .store import Session

__all__ = ["DiagnosticError", "Session", "__version__"]
