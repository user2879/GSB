"""Canonical serialization, hashing and overflow-safe integer helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_object(value: Any) -> str:
    return sha256_text(canonical_json(value))


def clamp_int64(value: int) -> int:
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return int(value)


def sat_add(*values: int) -> int:
    total = 0
    for value in values:
        total = clamp_int64(total + int(value))
    return total


def sat_mul(left: int, right: int) -> int:
    return clamp_int64(int(left) * int(right))


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncated toward zero, with an explicit zero guard."""

    if denominator == 0:
        return 0
    negative = (numerator < 0) != (denominator < 0)
    result = abs(int(numerator)) // abs(int(denominator))
    return -result if negative else result
