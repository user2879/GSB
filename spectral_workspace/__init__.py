"""Deterministic spectral material workspace.

The package is intentionally self contained: NumPy is the only third-party
runtime dependency, and all observer/adaptation tables are vendored.
"""

from .errors import DomainError

__all__ = ["DomainError"]
__version__ = "1.0.0"
