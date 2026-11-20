"""Shared deterministic helpers."""

from __future__ import annotations

import json
from typing import Any

# Frozen, stable key ordering so two identical decisions serialize byte-for-byte.
_CANONICAL_KW = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_dumps(payload: Any) -> str:
    """Serialize *payload* deterministically (sorted keys, no whitespace)."""
    return json.dumps(payload, **_CANONICAL_KW)


def canonical_bytes(payload: Any) -> bytes:
    return canonical_dumps(payload).encode("utf-8")


def is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division, explicit upward rounding, never float based."""
    if denominator <= 0:
        raise ValueError("ceil_div denominator must be positive")
    if numerator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def fixed_point(value: int, scale: int) -> str:
    """Render an integer fixed-point number (3 decimals for latency stats)."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    whole, frac = divmod(value, scale)
    return f"{sign}{whole}.{frac:0{len(str(scale)) - 1}d}"
