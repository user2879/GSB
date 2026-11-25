from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from . import FIXED_POINT_SCALE


SCALE = FIXED_POINT_SCALE


def as_q(value: Any, scale: int = SCALE) -> int:
    """Quantize a JSON number/string into a deterministic Q-format integer."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    if isinstance(value, int):
        return value * scale
    if isinstance(value, float):
        text = repr(value)
    elif isinstance(value, Decimal):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty numeric string")
    else:
        raise TypeError(f"unsupported numeric type: {type(value).__name__}")
    return int((Decimal(text) * scale).to_integral_value(rounding=ROUND_HALF_UP))


def q_from_int(value: int, scale: int = SCALE) -> int:
    return int(value) * scale


def q_mul(left: int, right: int, scale: int = SCALE) -> int:
    return (left * right) // scale


def q_dot(left: Iterable[int], right: Iterable[int], scale: int = SCALE) -> int:
    total = sum(int(a) * int(b) for a, b in zip(left, right))
    return total // scale


def divide_round(numerator: int, denominator: int) -> tuple[int, str]:
    """Round numerator/denominator to nearest int; ties go toward +infinity."""
    if denominator <= 0:
        raise ZeroDivisionError("denominator must be positive")
    floor_value = numerator // denominator
    exact = numerator % denominator == 0
    if exact:
        return floor_value, "exact"
    rounded = (2 * numerator + denominator) // (2 * denominator)
    ceil_value = -((-numerator) // denominator)
    if rounded == ceil_value:
        direction = "ceil"
    else:
        direction = "floor"
    return rounded, direction


def quantize_step(raw_q: int, step_q: int) -> tuple[int, int, str]:
    if step_q <= 0:
        raise ValueError("quantization step must be positive")
    index, direction = divide_round(raw_q, step_q)
    return index * step_q, index, direction


def mean_int(values: list[int]) -> int:
    if not values:
        return 0
    return sum(values) // len(values)


def stable_range(values: list[int]) -> int:
    if not values:
        return 0
    return max(values) - min(values)
