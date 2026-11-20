"""Deterministic fixed-point quantization for integer and float values.

All quantization uses explicit integer arithmetic so results are identical on
every platform, Python build and partition count.
"""

from decimal import Decimal, ROUND_HALF_UP

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NULL_Q = INT64_MIN


def validate_fraction_bits(fraction_bits):
    if not isinstance(fraction_bits, int) or isinstance(fraction_bits, bool):
        raise ValueError("fraction_bits must be an integer")
    if not 0 <= fraction_bits <= 30:
        raise ValueError("fraction_bits must be within [0, 30]")
    return fraction_bits


def scale_factor(fraction_bits):
    validate_fraction_bits(fraction_bits)
    return 1 << fraction_bits


def quantize(value, fraction_bits):
    """Quantize an int/float to a signed 64-bit fixed-point integer.

    Rounding is half away from zero (decimal ROUND_HALF_UP) to keep the rule
    explicit and reproducible. ``None`` maps to the reserved :data:`NULL_Q`.
    """
    if value is None:
        return NULL_Q
    scale = scale_factor(fraction_bits)
    if isinstance(value, bool):
        raise ValueError("boolean values cannot be quantized")
    if isinstance(value, int):
        q = value * scale
    elif isinstance(value, float):
        decimal_value = Decimal(repr(value)) * scale
        q = int(decimal_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    else:
        raise ValueError(f"cannot quantize value of type {type(value).__name__}")
    if not INT64_MIN < q <= INT64_MAX:
        raise ValueError(f"quantized value {q} overflows signed 64-bit range")
    return q


def dequantize(q, fraction_bits):
    """Inverse of :func:`quantize` for non-null quantized integers."""
    if q == NULL_Q:
        return None
    scale = scale_factor(fraction_bits)
    return q / scale


def quantization_error(value, fraction_bits):
    """Signed float reference error: fixed_point_value - float_reference."""
    if value is None:
        return 0.0
    return dequantize(quantize(value, fraction_bits), fraction_bits) - float(value)


def error_bound(fraction_bits):
    """Worst-case absolute quantization error per single float value."""
    if fraction_bits <= 0:
        return 0.5
    return 1.0 / (2 * scale_factor(fraction_bits))


def sum_error_bound(count, fraction_bits):
    """Worst-case absolute error of summing ``count`` quantized values."""
    return count * error_bound(fraction_bits)
