"""Deterministic signed Q(F) fixed-point helpers.

All values are signed integers with an implicit scale of ``2**F``.  Arithmetic
is exact integer arithmetic; rounding only happens at explicit rescale points
so the pipeline is reproducible across platforms and thread schedules.
"""

from __future__ import annotations

import numpy as np

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
ROUNDING_MODES = ("truncate", "half_up", "half_even")


def _round_scaled(abs_num: int, den: int, rounding: str) -> int:
    quotient, remainder = divmod(abs_num, den)
    twice = remainder * 2
    if rounding == "truncate":
        return quotient
    if rounding == "half_up":
        if twice >= den:
            quotient += 1
        return quotient
    if rounding == "half_even":
        if twice > den or (twice == den and quotient % 2 == 1):
            quotient += 1
        return quotient
    raise ValueError(f"unsupported rounding mode: {rounding}")


def rounded_div(num: int, den: int, rounding: str) -> int:
    """Signed integer division with configurable nearest/zero rounding."""
    if den <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if num < 0 else 1
    return sign * _round_scaled(abs(num), den, rounding)


def rescale_array(values: np.ndarray, f_from: int, f_to: int, rounding: str) -> np.ndarray:
    """Rescale an integer Q(f_from) array to Q(f_to)."""
    values = np.asarray(values, dtype=np.int64)
    if f_from == f_to:
        return values.copy()
    if f_from < f_to:
        return values << (f_to - f_from)
    shift = f_from - f_to
    den = 1 << shift
    abs_values = np.abs(values, dtype=np.int64)
    quotient = abs_values // den
    remainder = abs_values % den
    twice = remainder * 2
    if rounding == "truncate":
        increment = np.zeros_like(quotient)
    elif rounding == "half_up":
        increment = (twice >= den).astype(np.int64)
    elif rounding == "half_even":
        increment = ((twice > den) | ((twice == den) & (quotient % 2 == 1))).astype(np.int64)
    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    magnitude = quotient + increment
    return np.where(values < 0, -magnitude, magnitude).astype(np.int64)


def narrow(values: np.ndarray, f_from: int, f_to: int, rounding: str) -> np.ndarray:
    if f_to > f_from:
        raise ValueError("narrow requires f_to <= f_from")
    return rescale_array(values, f_from, f_to, rounding)


def widen(values: np.ndarray, f_from: int, f_to: int) -> np.ndarray:
    if f_to < f_from:
        raise ValueError("widen requires f_to >= f_from")
    return rescale_array(values, f_from, f_to, "truncate")


def qmul(a: np.ndarray, b: np.ndarray, f_bits: int, rounding: str) -> np.ndarray:
    """Multiply two Q(F) integer arrays, returning Q(F)."""
    product = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    den = 1 << f_bits
    abs_product = np.abs(product, dtype=np.int64)
    quotient = abs_product // den
    remainder = abs_product % den
    twice = remainder * 2
    if rounding == "truncate":
        increment = np.zeros_like(quotient)
    elif rounding == "half_up":
        increment = (twice >= den).astype(np.int64)
    elif rounding == "half_even":
        increment = ((twice > den) | ((twice == den) & (quotient % 2 == 1))).astype(np.int64)
    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    magnitude = quotient + increment
    return np.where(product < 0, -magnitude, magnitude).astype(np.int64)


def qdiv(num: np.ndarray, den: int, f_bits: int, rounding: str) -> np.ndarray:
    """Divide a Q(F) numerator array by a Q(F) integer denominator."""
    if den <= 0:
        raise ValueError("denominator must be positive")
    numerator = np.asarray(num, dtype=np.int64)
    abs_num = np.abs(numerator, dtype=np.int64)
    quotient = abs_num // den
    remainder = abs_num % den
    twice = remainder * 2
    if rounding == "truncate":
        increment = np.zeros_like(quotient)
    elif rounding == "half_up":
        increment = (twice >= den).astype(np.int64)
    elif rounding == "half_even":
        increment = ((twice > den) | ((twice == den) & (quotient % 2 == 1))).astype(np.int64)
    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    magnitude = quotient + increment
    return np.where(numerator < 0, -magnitude, magnitude).astype(np.int64)


def saturate_i32(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.int64), INT32_MIN, INT32_MAX).astype(np.int32)


def quantize_float(value: float, f_bits: int, rounding: str) -> int:
    """Quantize a real number to a Q(F) integer."""
    scaled = value * (1 << f_bits)
    if rounding == "truncate":
        return int(np.trunc(scaled))
    if rounding == "half_up":
        return int(np.floor(scaled + 0.5)) if scaled >= 0 else -int(np.floor(-scaled + 0.5))
    if rounding == "half_even":
        return int(np.rint(scaled))
    raise ValueError(f"unsupported rounding mode: {rounding}")


def to_uint8(values: np.ndarray, f_bits: int, rounding: str) -> np.ndarray:
    """Convert a Q(F) array (treated as [0, 1] range) to uint8 bytes."""
    clipped = np.clip(np.asarray(values, dtype=np.int64), 0, (1 << f_bits) - 1)
    if f_bits >= 8:
        scaled = narrow(clipped, f_bits, 8, rounding)
    else:
        scaled = widen(clipped, f_bits, 8)
    return np.clip(scaled, 0, 255).astype(np.uint8)
