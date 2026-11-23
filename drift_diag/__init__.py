"""Declarative configuration drift diagnostic component.

The package is deliberately dependency-free and never mutates a real cluster.
It consumes local JSON/NDJSON fixtures, persists an auditable SQLite ledger,
and offers a read-only Tkinter diagnostic view.
"""

from .errors import DiagnosticError

__all__ = ["DiagnosticError"]
__version__ = "0.1.0"
