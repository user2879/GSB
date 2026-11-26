"""Decimal-backed deterministic fixed-point quantization and integer rounding."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any


def as_decimal(value: Any, field: str = "value") -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not boolean")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        result = Decimal(str(value))
    elif isinstance(value, str):
        result = Decimal(value)
    else:
        raise ValueError(f"{field} must be a finite number")
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def round_half_away_integer(value: Decimal) -> int:
    """Round to nearest integer; ties move away from zero."""

    with localcontext() as ctx:
        ctx.prec = 80
        return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def round_div(num: int, den: int) -> tuple[int, int, str]:
    """Round ``num / den`` to int with half-away ties and exact remainder.

    The returned remainder follows the numerator sign. ``direction`` is one of
    exact/down/up/half_down/half_up where up means away from zero.
    """

    if den == 0:
        raise ZeroDivisionError("round_div requires a nonzero denominator")
    if den < 0:
        num = -num
        den = -den
    sign = -1 if num < 0 else 1
    abs_num = abs(num)
    quotient, remainder = divmod(abs_num, den)
    twice = remainder * 2
    if remainder == 0:
        rounded = sign * quotient
        direction = "exact"
    elif twice < den:
        rounded = sign * quotient
        direction = "down" if sign > 0 else "toward_zero"
    elif twice > den:
        quotient += 1
        rounded = sign * quotient
        direction = "up"
    else:
        quotient += 1
        rounded = sign * quotient
        direction = "half_up"
    signed_remainder = sign * remainder
    return rounded, signed_remainder, direction


@dataclass(frozen=True)
class FixedParams:
    decimal_places: int = 9
    quant_level: int = 1
    accum_rounding_margin: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.decimal_places, int) or isinstance(self.decimal_places, bool):
            raise ValueError("decimal_places must be an integer")
        if not (0 <= self.decimal_places <= 18):
            raise ValueError("decimal_places must be in [0, 18]")
        if not isinstance(self.quant_level, int) or isinstance(self.quant_level, bool):
            raise ValueError("quant_level must be an integer")
        if not (0 <= self.quant_level <= self.decimal_places):
            raise ValueError("quant_level must be in [0, decimal_places]")
        if not isinstance(self.accum_rounding_margin, int) or isinstance(self.accum_rounding_margin, bool):
            raise ValueError("accum_rounding_margin must be an integer")
        if not (0 <= self.accum_rounding_margin <= 3 * self.decimal_places):
            raise ValueError("accum_rounding_margin must be in [0, 3*decimal_places]")

    @property
    def scale(self) -> int:
        return 10**self.decimal_places

    @property
    def grid(self) -> int:
        return 10**self.quant_level

    @property
    def accum_digits(self) -> int:
        return max(0, 3 * self.decimal_places - self.accum_rounding_margin)

    @property
    def accum_scale(self) -> int:
        return 10**self.accum_digits

    @property
    def accum_divisor(self) -> int:
        return 10**self.accum_rounding_margin

    @property
    def half_zero_units(self) -> Decimal:
        return Decimal(self.grid) / Decimal(2)

    def fingerprint_material(self) -> dict[str, Any]:
        return {
            "decimal_places": self.decimal_places,
            "quant_level": self.quant_level,
            "accum_rounding_margin": self.accum_rounding_margin,
            "scale": str(self.scale),
            "grid": str(self.grid),
            "accum_digits": self.accum_digits,
        }


@dataclass
class QuantizedValue:
    integer: int
    raw_decimal: Decimal
    residual: Decimal
    step: Decimal
    tie: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "integer": self.integer,
            "raw": format(self.raw_decimal, "f"),
            "quantized": format(Decimal(self.integer) / Decimal(10) ** _places_from_step(self.step), "f"),
            "residual": format(self.residual, "f"),
            "step": format(self.step, "f"),
            "tie": self.tie,
        }


def _places_from_step(step: Decimal) -> int:
    exponent = step.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def quantize_fine(value: Decimal, params: FixedParams) -> tuple[int, Decimal]:
    """Round a scalar to the finest fixed-point scale."""

    scaled = value * Decimal(params.scale)
    integer = round_half_away_integer(scaled)
    quantized = Decimal(integer) / Decimal(params.scale)
    return integer, value - quantized


def quantize_feature(value: Decimal, params: FixedParams) -> QuantizedValue:
    """Quantize directly to the effective grid ``10**(quant_level-decimal_places)``."""

    grid = params.grid
    coarse_scale = params.scale // grid
    scaled = value * Decimal(coarse_scale)
    coarse_integer = round_half_away_integer(scaled)
    integer = coarse_integer * grid
    scale_decimal = Decimal(params.scale)
    quantized = Decimal(integer) / scale_decimal
    residual = value - quantized
    step = Decimal(grid) / scale_decimal
    tie = abs(scaled - Decimal(coarse_integer)) == Decimal("0.5")
    return QuantizedValue(
        integer=integer,
        raw_decimal=value,
        residual=residual,
        step=step,
        tie=tie,
    )


def fixed_to_float(integer: int, params: FixedParams) -> float:
    return float(Decimal(integer) / Decimal(params.scale))
