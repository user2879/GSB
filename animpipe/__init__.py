"""Deterministic skeletal-animation retargeting pipeline.

All public entry points are deterministic: given the same inputs and
configuration, byte-identical content hashes are produced regardless of
thread count or frame-block partitioning.
"""

from .errors import (
    AnimError,
    E_SCHEMA_VERSION,
    E_QUAT_INVALID,
    E_INDEX_OVERFLOW,
    E_TIME_NONMONOTONIC,
    E_RETARGET_DIVERGE,
    E_BUDGET_DENIED,
    E_CLOSURE_MISMATCH,
)

__all__ = [
    "AnimError",
    "E_SCHEMA_VERSION",
    "E_QUAT_INVALID",
    "E_INDEX_OVERFLOW",
    "E_TIME_NONMONOTONIC",
    "E_RETARGET_DIVERGE",
    "E_BUDGET_DENIED",
    "E_CLOSURE_MISMATCH",
]
