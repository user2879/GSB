"""Integer-only helpers for deterministic Q-format arithmetic."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def q_scale(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    return 1 << bits


def div_round_nearest(numerator: int, denominator: int) -> tuple[int, str, int]:
    """Round an integer ratio using half-to-even and report the rounding direction."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator == 0:
        return 0, "exact", 0
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    if remainder == 0:
        return sign * quotient, "exact", 0
    twice = remainder * 2
    if twice < denominator:
        return sign * quotient, "toward_zero", remainder
    if twice > denominator:
        return sign * (quotient + 1), "away_zero", remainder
    if quotient % 2 == 0:
        return sign * quotient, "half_even_down", remainder
    return sign * (quotient + 1), "half_even_up", remainder


def quantize(value: int, step: int) -> tuple[int, str, int]:
    if step <= 0:
        raise ValueError("step must be positive")
    quotient, direction, remainder = div_round_nearest(value, step)
    return quotient * step, direction, remainder


def determinant3(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> int:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def matmul3(
    left: list[list[int]] | tuple[tuple[int, ...], ...],
    right: list[list[int]] | tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int, int], ...]:
    rows = []
    for row in range(3):
        rows.append(
            tuple(
                sum(left[row][k] * right[k][column] for k in range(3))
                for column in range(3)
            )
        )
    return tuple(rows)


def transpose3(
    matrix: list[list[int]] | tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def round_table(value: tuple[int, str, int] | tuple[int, str]) -> dict[str, int | str]:
    return {"value": value[0], "direction": value[1], "remainder": value[2]}


def padded(value: int, width: int = 8) -> str:
    return f"{value:0{width}d}"
