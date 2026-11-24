"""Deterministic clustering and density workbench.

The public API is intentionally small.  ``Workbench`` implements the interactive
command loop, while ``execute_pipeline`` is the synchronous canonical path used
for full-rebuild equivalence checks.
"""

from .config import load_config, normalize_config
from .data import generate_table, load_data_description
from .errors import Issue
from .pipeline import execute_pipeline
from .workbench import Workbench

__all__ = [
    "Issue",
    "Workbench",
    "execute_pipeline",
    "generate_table",
    "load_config",
    "load_data_description",
    "normalize_config",
]

__version__ = "1.0.0"
