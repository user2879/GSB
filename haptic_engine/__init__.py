"""Local haptic command-stream synthesis engine.

The package turns a deterministic haptic event script into per-device driver
command streams while respecting per-device frame bandwidth and a global
in-flight buffer budget.
"""

from .errors import HapticError

__all__ = ["HapticError"]
