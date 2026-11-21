"""Distributed replica-manifest interoperability and local repair component."""

from .errors import RepairError
from .model import CURRENT_MANIFEST_VERSION

__all__ = ["CURRENT_MANIFEST_VERSION", "RepairError"]
__version__ = "1.0.0"
