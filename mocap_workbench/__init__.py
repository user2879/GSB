"""Deterministic motion-capture cleaning workbench.

The package separates deterministic domain logic (generation, detection,
repair, evaluation and incremental dependency handling) from the PySide6
desktop shell so the complete business pipeline can be verified headlessly.
"""

from .constants import APP_NAME, APP_VERSION, FPS, FRAME_COUNTS, SKELETON
from .errors import BusinessError, ErrorCode

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "BusinessError",
    "ErrorCode",
    "FPS",
    "FRAME_COUNTS",
    "SKELETON",
]
