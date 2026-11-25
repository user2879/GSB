"""Deterministic integer and fixed-point helpers.

All coordinates consumed by the reconstruction engine are Python integers in
the Q(fixed_bits) domain.  No float is used on a reconstruction path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def round_div(numerator: int, denominator: int) -> tuple[int, str]:
    """Round n/d to nearest integer, ties toward +infinity.

    The returned direction describes the chosen neighbour of floor(n/d): one of
    ``exact``, ``up`` or ``down``.  A negative denominator is normalised first.
    """

    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    floor_q, remainder = divmod(numerator, denominator)
    if remainder == 0:
        return floor_q, "exact"
    if remainder * 2 < denominator:
        return floor_q, "down"
    return floor_q + 1, "up"


def rounded(numerator: int, denominator: int) -> int:
    return round_div(numerator, denominator)[0]


def quantize(value: int, quantum: int) -> tuple[int, str]:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    units, direction = round_div(value, quantum)
    return units * quantum, direction


def quantize_xyz(xyz: Iterable[int], quantum: int) -> tuple[list[int], dict[str, str]]:
    values = list(xyz)
    if len(values) != 3:
        raise ValueError("xyz must have three components")
    out: list[int] = []
    directions: dict[str, str] = {}
    for axis, value in zip(("x", "y", "z"), values):
        q_value, direction = quantize(value, quantum)
        out.append(q_value)
        directions[axis] = direction
    return out, directions


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(value: str, stage: str) -> str:
    return f"{stage}:{value}"


def mean_int(values: Iterable[int]) -> int:
    values_list = list(values)
    if not values_list:
        return 0
    return rounded(sum(values_list), len(values_list))
