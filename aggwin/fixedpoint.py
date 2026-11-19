"""Fixed-point integer arithmetic with explicit truncation direction.

All aggregation state is kept as scaled integers: scaled = value * scale.
The mean is trunc(sum_scaled / count) under the session truncation
direction.  Quantisation error bound: |reported/scale - true| <= 1/scale.
"""

from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR

DIRECTIONS = ("floor", "ceil", "toward_zero")
_ROUNDING = {"floor": ROUND_FLOOR, "ceil": ROUND_CEILING,
             "toward_zero": ROUND_DOWN}


def to_scaled(value, scale, direction):
    """Convert an int/float to a scaled integer under `direction`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"non-numeric value {value!r}")
    if isinstance(value, int):
        return value * scale
    dec = Decimal(repr(value)) * scale
    return int(dec.quantize(Decimal("1"), rounding=_ROUNDING[direction]))


def trunc_div(numer, denom, direction):
    """Integer division of numer/denom truncated per direction."""
    if denom == 0:
        raise ZeroDivisionError("trunc_div by zero")
    if direction == "floor":
        return numer // denom
    if direction == "ceil":
        return -((-numer) // denom)
    # toward_zero
    quotient = abs(numer) // abs(denom)
    return quotient if (numer >= 0) == (denom >= 0) else -quotient


def mean_scaled(sum_scaled, count, direction):
    if count == 0:
        return 0
    return trunc_div(sum_scaled, count, direction)


def mean_error_bound(scale):
    """Upper bound of |reported_mean/scale - true_mean|."""
    return Decimal(1) / Decimal(scale)
