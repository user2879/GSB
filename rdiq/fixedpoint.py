"""Deterministic fixed-point integer arithmetic with explicit rules.

All confidence values live in a signed two's-complement fixed-point domain.
The least significant bit represents ``1 / scale`` of a unit confidence, so
one ULP equals 1 in integer representation.

Explicit arithmetic contract used everywhere in the package:

* Multiplication of two W-bit operands is performed in a signed 2W-bit
  accumulator and saturated to the 2W-bit signed range.
* Summation uses the same 2W-bit accumulator and saturates on overflow.
* Every division truncates toward zero (``q = sign * (abs(a) // abs(b))``);
  the discarded magnitude is recorded as the truncation residual.
* Final means and views are narrowed back to the W-bit range by saturation.
* Weights are quantized once by truncating toward zero, then clamped to the
  W-bit signed range; the discarded part is the source-level quantization
  residual.
"""

from .errors import SpecError

FORMAT_VERSION = 1
WEIGHT_BASE = 10000
SUPPORTED_WIDTHS = (8, 16, 32)
SUPPORTED_SCALES = (256, 1024, 4096)


def int_min(width):
    return -(2 ** (width - 1))


def int_max(width):
    return 2 ** (width - 1) - 1


def acc_min(width):
    return -(2 ** (2 * width - 1))


def acc_max(width):
    return 2 ** (2 * width - 1) - 1


def clamp(value, width):
    """Saturate ``value`` to the signed W-bit range; report saturation."""
    low, high = int_min(width), int_max(width)
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def clamp_acc(value, width):
    """Saturate ``value`` to the signed 2W-bit accumulator range."""
    low, high = acc_min(width), acc_max(width)
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def sat_add(left, right, width):
    total = left + right
    value, saturated = clamp_acc(total, width)
    return value, saturated


def mul_wide(left, right, width):
    product = left * right
    return clamp_acc(product, width)


def div_trunc(numerator, denominator):
    """Integer division with truncation toward zero and residual."""
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        quotient = -quotient
    remainder = numerator - quotient * denominator
    return quotient, remainder


def quantize_weight(raw_weight, scale, width):
    """Quantize a non-negative integer raw weight (basis 10000) to Q units."""
    if not isinstance(raw_weight, int) or raw_weight < 0:
        raise SpecError(
            "raw weight must be a non-negative integer",
            field="raw_weight",
            raw_weight=raw_weight,
        )
    numerator = raw_weight * scale
    weight, residual = div_trunc(numerator, WEIGHT_BASE)
    weight, saturated = clamp(weight, width)
    return {
        "raw_weight": raw_weight,
        "quantized": weight,
        "residual": abs(residual),
        "saturated": saturated,
    }


def validate_width(width):
    if not isinstance(width, int) or isinstance(width, bool) or width not in SUPPORTED_WIDTHS:
        raise SpecError(
            "unsupported fixed-point width",
            field="width",
            value=width,
            supported=list(SUPPORTED_WIDTHS),
        )


def validate_scale(scale):
    if not isinstance(scale, int) or isinstance(scale, bool) or scale not in SUPPORTED_SCALES:
        raise SpecError(
            "unsupported scale factor",
            field="scale",
            value=scale,
            supported=list(SUPPORTED_SCALES),
        )
