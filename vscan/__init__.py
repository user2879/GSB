"""Deterministic, vectorized local columnar scan engine."""

from .errors import VScanError
from .session import Session

__all__ = ["Session", "VScanError"]
__version__ = "0.1.0"
