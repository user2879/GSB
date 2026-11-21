from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN


def quantization_step(quantization_level: int) -> int:
    if quantization_level < 0:
        raise ValueError("quantization_level must be non-negative")
    return 1 << int(quantization_level)


def quantize_number(value: object, fixed_decimals: int, quantization_level: int) -> int:
    """Quantize a real number to a signed integer at the configured fixed scale.

    The raw fixed-point integer is rounded half-to-even, then snapped to a
    multiple of ``2 ** quantization_level``.  All downstream arithmetic is
    integer-only.
    """

    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite weight")
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError("non-finite weight")
    scale = 10 ** int(fixed_decimals)
    exact = (decimal_value * scale).to_integral_value(rounding=ROUND_HALF_EVEN)
    raw = int(exact)
    step = quantization_step(quantization_level)
    if step == 1:
        return raw
    sign = -1 if raw < 0 else 1
    magnitude = abs(raw)
    quotient, remainder = divmod(magnitude, step)
    if remainder * 2 > step or (remainder * 2 == step and quotient % 2 == 1):
        quotient += 1
    return sign * quotient * step


def round_div_half_even(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(int(numerator)), int(denominator))
    if remainder * 2 > denominator or (remainder * 2 == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def fixed_mul_div(left: int, right: int, scale: int) -> int:
    return round_div_half_even(int(left) * int(right), int(scale))


def fixed_scale(fixed_decimals: int) -> int:
    return 10 ** int(fixed_decimals)


def fixed_to_float(value: int, scale: int) -> float:
    return float(value) / float(scale)
