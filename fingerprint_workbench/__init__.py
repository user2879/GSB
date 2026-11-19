"""Deterministic audio fingerprint workbench."""

from .errors import WorkbenchError
from .workbench import Workbench

__all__ = ["Workbench", "WorkbenchError"]
__version__ = "0.1.0"
