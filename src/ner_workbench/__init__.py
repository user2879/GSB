"""Deterministic NER parallel scheduling and resource diagnostics workbench."""

from .errors import WorkbenchError
from .runner import Workbench, compare_reports, load_report

__all__ = ["Workbench", "WorkbenchError", "compare_reports", "load_report"]

__version__ = "0.1.0"
