"""Deterministic columnar page-cache and spill engine.

The public modules deliberately keep the on-disk formats, cache policy, and
fixed-point aggregation rules explicit so that runs are reproducible across
process restarts and Windows/Linux file systems.
"""

from .errors import EngineError, ErrorCode

__all__ = ["EngineError", "ErrorCode"]
__version__ = "1.0.0"
