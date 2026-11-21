"""Deterministic local model-configuration dispatch workbench."""

from .errors import WorkbenchError
from .spec import load_spec, validate_spec

__all__ = ["WorkbenchError", "load_spec", "validate_spec"]
__version__ = "1.0.0"
