from __future__ import annotations

INT64_MAX = (1 << 63) - 1


def floor_div(numerator: int, denominator: int) -> int:
    """Explicit truncation rule: floor for non-negative fixed-point values."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


def saturate(value: int, limit: int = INT64_MAX) -> int:
    """Overflow rule: clamp to [0, limit] instead of wrapping."""

    if value < 0:
        return 0
    if value > limit:
        return limit
    return value


def scaled_ms_to_ceil(value_scaled: int, scale: int) -> int:
    return ceil_div(max(0, value_scaled), scale)
