"""Exact fixed-point integer arithmetic used by every scoring stage.

All domain quantities (weights, role credibility, annotation confidence,
deduction/bonus strength, thresholds, scores) are stored as per-mille
integers (``x`` means ``x / 1000``).  The pipeline never performs binary
floating point aggregation: division uses decimal half-up rounding so that
the result is reproducible byte-for-byte.
"""

from decimal import ROUND_HALF_UP, Decimal, localcontext

TAG_INT_PERMILLE = "INT_PERMILLE"
TAG_FLOAT_APPROX = "FLOAT_APPROX"

MAX_DECIMAL_WIDTH = 9


def round_half_up_div(numerator: int, denominator: int) -> int:
    """Integer half-up division, correct for negative numerators."""
    if denominator == 0:
        raise ZeroDivisionError("round_half_up_div denominator is zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    q, r = divmod(abs(numerator), abs(denominator))
    if 2 * r >= abs(denominator):
        q += 1
    return sign * q


def mul_permille(value: int, factor_permille: int) -> int:
    """Multiply a fixed-point integer by a per-mille factor, half-up."""
    return round_half_up_div(value * factor_permille, 1000)


def scale_factor(value_permille: int, decimals: int) -> int:
    """Scale a per-mille quantity to a fixed-point grid with ``decimals``."""
    scale = 10 ** decimals
    return round_half_up_div(value_permille * scale, 1000)


def quantize(score: int, quantum: int) -> tuple:
    """Snap ``score`` to a multiple of positive ``quantum`` (half-up).

    Returns ``(quantized_score, residual)`` where residual is the signed
    truncation remainder (``score - quantized_score``).
    """
    if quantum <= 0:
        return score, 0
    q = round_half_up_div(score, quantum) * quantum
    return q, score - q


def decay_coeff(delta_ms: int, half_life_ms: int) -> int:
    """Per-mille exponential half-life coefficient, half-up rounded.

    ``1000 * 0.5 ** (delta_ms / half_life_ms)`` computed in decimal so
    changing the half-life changes the coefficient, not just its display.
    """
    if half_life_ms <= 0:
        raise ValueError("half_life_ms must be positive")
    if delta_ms == 0:
        return 1000
    with localcontext() as ctx:
        ctx.prec = 50
        exponent = Decimal(delta_ms) / Decimal(half_life_ms)
        raw = Decimal(1000) * (Decimal(1) / Decimal(2)) ** exponent
        return int(raw.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def clamp_low(value: int, floor: int) -> int:
    return value if value >= floor else floor


def permille_to_float(value_permille: int) -> float:
    """Binary-float view of a per-mille integer (display only)."""
    return value_permille / 1000.0


def float_to_permille(value_float: float) -> int:
    """Exact per-mille integer from a float, via its shortest decimal text."""
    with localcontext() as ctx:
        ctx.prec = 50
        dec = Decimal(str(value_float)) * Decimal(1000)
        return int(dec.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def validate_decimal_width(decimals: int) -> None:
    if not isinstance(decimals, int) or isinstance(decimals, bool):
        raise ValueError("decimal width must be an integer")
    if decimals < 0 or decimals > MAX_DECIMAL_WIDTH:
        raise ValueError(
            f"decimal width {decimals} out of range [0, {MAX_DECIMAL_WIDTH}]"
        )
