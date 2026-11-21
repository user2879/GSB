"""Fixed-point integer arithmetic and decimal-grid quantization.

All routing/recalibration math is carried out as Python integers on a binary
fixed-point scale (Q16 by default).  Conversions from JSON decimals go through
:mod:`decimal` so floating-point text never introduces rounding ambiguity.

Quantization levels are *decimal grids* (``grid`` = smallest representable
step as a decimal string).  Switching a level therefore changes both the raw
and quantized outputs whenever a value does not land exactly on that grid.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_HALF_EVEN

ONE = 1
DEFAULT_SCALE_BITS = 16


def scale_factor(scale_bits: int = DEFAULT_SCALE_BITS) -> int:
    return 1 << scale_bits


def decimal_to_fixed(value, scale_bits: int = DEFAULT_SCALE_BITS) -> int:
    """Convert a JSON-decoded number/string to a fixed-point integer."""
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError("boolean is not a fixed-point number")
    if isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite value cannot be fixed-point encoded")
        dec = Decimal(repr(value))
    elif isinstance(value, str):
        dec = Decimal(value)
    elif isinstance(value, Decimal):
        dec = value
    else:
        raise TypeError(f"unsupported fixed-point source: {type(value)!r}")
    if not dec.is_finite():
        raise ValueError("non-finite value cannot be fixed-point encoded")
    with localcontext() as ctx:
        ctx.prec = 50
        scaled = dec * Decimal(scale_factor(scale_bits))
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def fixed_to_decimal(value: int, scale_bits: int = DEFAULT_SCALE_BITS) -> Decimal:
    return Decimal(value) / Decimal(scale_factor(scale_bits))


def fixed_to_str(value: int, scale_bits: int = DEFAULT_SCALE_BITS) -> str:
    """Canonical shortest decimal string for a binary fixed-point integer."""
    dec = fixed_to_decimal(value, scale_bits)
    text = format(dec, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def fixed_div(numerator: int, denominator: int) -> int:
    """Deterministic floor division on fixed-point integers."""
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def clamp_fixed(value: int, scale_bits: int = DEFAULT_SCALE_BITS) -> int:
    factor = scale_factor(scale_bits)
    if value < 0:
        return 0
    if value > factor:
        return factor
    return value


def quantize_to_grid(value: int, grid: str, scale_bits: int = DEFAULT_SCALE_BITS) -> int:
    """Snap a fixed-point value to the nearest multiple of a decimal grid."""
    grid_fixed = decimal_to_fixed(grid, scale_bits)
    if grid_fixed <= 0:
        raise ValueError("quantization grid must be positive")
    q, r = divmod(value, grid_fixed)
    if r * 2 >= grid_fixed:
        q += 1
    return q * grid_fixed


def is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, Decimal)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, str):
        try:
            return Decimal(value).is_finite()
        except Exception:
            return False
    return False
