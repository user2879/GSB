"""Stable JSON and content fingerprints used by cache keys and reports."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def stable_json(value: Any, *, pretty: bool = False) -> str:
    """Serialize a value without host, ordering, or NaN-dependent variation."""
    if pretty:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return value[:length]


def stable_float(value: float, digits: int = 12) -> float:
    """Round finite floats and normalize negative zero for JSON determinism."""
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    rounded = round(float(value), digits)
    if rounded == 0:
        return 0.0
    return rounded


def stable_vector(values: list[float] | tuple[float, ...], digits: int = 12) -> list[float]:
    return [stable_float(value, digits) for value in values]
