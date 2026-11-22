"""Deterministic image-edit exchange workbench."""

from .config import WorkbenchConfig, load_config
from .models import RegionEdit, Rect

__all__ = ["RegionEdit", "Rect", "WorkbenchConfig", "load_config"]
__version__ = "0.1.0"

