"""Canonical JSON, stable floats, and hashing.

Every persisted record and every hash goes through this module.  Numeric
rounding removes platform-dependent last-bit differences while retaining far
more precision than is meaningful for colour measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_DECIMALS = 12


def stable_float(value: float | np.floating[Any], decimals: int = DEFAULT_DECIMALS) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number cannot be serialized: {value!r}")
    rounded = round(value, decimals)
    # Avoid the two spellings of zero and accidental negative zero in reports.
    if rounded == 0:
        return 0.0
    return rounded


def canonicalize(value: Any, decimals: int = DEFAULT_DECIMALS) -> Any:
    if isinstance(value, np.ndarray):
        return [canonicalize(item, decimals) for item in value.tolist()]
    if isinstance(value, np.generic):
        return canonicalize(value.item(), decimals)
    if isinstance(value, (float, np.floating)):
        return stable_float(float(value), decimals)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item, decimals)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, decimals) for item in value]
    if isinstance(value, set):
        return [canonicalize(item, decimals) for item in sorted(value, key=str)]
    return value


def canonical_json(value: Any, decimals: int = DEFAULT_DECIMALS) -> str:
    return json.dumps(
        canonicalize(value, decimals),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any, decimals: int = DEFAULT_DECIMALS) -> bytes:
    return (canonical_json(value, decimals) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(value: Any, decimals: int = DEFAULT_DECIMALS) -> str:
    return sha256_bytes(canonical_bytes(value, decimals))


def array_digest(values: Sequence[Sequence[float]] | np.ndarray) -> str:
    return sha256_obj(np.asarray(values, dtype=float).tolist())
