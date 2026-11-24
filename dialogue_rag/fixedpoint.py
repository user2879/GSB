"""Deterministic fixed-point profiles and exact reference arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any


ALLOWED_WIDTHS = (8, 16, 32)
ALLOWED_GEARS = ("G0", "G1", "G2", "G3")
GEAR_LOSS_BITS = {"G0": 0, "G1": 2, "G2": 4, "G3": 6}
_GEAR_ALIASES = {
    "Q0": "G0",
    "Q1": "G1",
    "Q2": "G2",
    "Q3": "G3",
    "fine": "G0",
    "standard": "G1",
    "balanced": "G1",
    "coarse": "G2",
    "draft": "G3",
}


def parse_width(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a width")
    if isinstance(value, int):
        width = value
    else:
        text = str(value).strip().upper()
        if text.startswith("F"):
            text = text[1:]
        width = int(text)
    if width not in ALLOWED_WIDTHS:
        raise ValueError(f"fixed_width must be one of {ALLOWED_WIDTHS}")
    return width


def parse_gear(value: Any) -> str:
    text = str(value).strip()
    normalized = _GEAR_ALIASES.get(text, _GEAR_ALIASES.get(text.lower(), text.upper()))
    if normalized not in ALLOWED_GEARS:
        raise ValueError(f"quant_gear must be one of {ALLOWED_GEARS}")
    return normalized


@dataclass(frozen=True)
class FixedProfile:
    width: int = 16
    gear: str = "G1"
    rounding: str = "half_up"

    @property
    def scale(self) -> int:
        return 1 << self.width

    @property
    def gear_step(self) -> int:
        return 1 << GEAR_LOSS_BITS[self.gear]

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {
            "fixed_width": f"F{self.width}",
            "quant_gear": self.gear,
            "rounding": self.rounding,
            "scale": self.scale,
            "gear_step": self.gear_step,
        }

    def quantize(self, value: Fraction | Decimal | int | float) -> int:
        fraction = as_fraction(value)
        numerator = fraction.numerator * self.scale
        denominator = fraction.denominator
        if numerator >= 0:
            rounded = (2 * numerator + denominator) // (2 * denominator)
        else:
            rounded = -((-2 * numerator + denominator) // (2 * denominator))
        return quantize_to_grid(rounded, self.gear_step)

    def multiply(self, left_int: int, right_int: int) -> int:
        product = left_int * right_int
        rounded = half_up_div(product, self.scale)
        return quantize_to_grid(rounded, self.gear_step)

    def decode(self, value_int: int) -> float:
        return value_int / self.scale

    def decimal_text(self, value_int: int, digits: int | None = None) -> str:
        if digits is None:
            digits = max(6, self.width // 4)
        return f"{self.decode(value_int):.{digits}f}"


def quantize_to_grid(value: int, step: int) -> int:
    if step <= 1:
        return value
    if value >= 0:
        return ((value + step // 2) // step) * step
    return -(((-value + step // 2) // step) * step)


def half_up_div(numerator: int, denominator: int) -> int:
    if numerator >= 0:
        return (2 * numerator + denominator) // (2 * denominator)
    positive_numerator = -numerator
    return -((2 * positive_numerator + denominator) // (2 * denominator))


def as_fraction(value: Fraction | Decimal | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(Decimal(repr(value)))
    return Fraction(Decimal(str(value)))


def fraction_float(value: Fraction, digits: int = 12) -> float:
    return round(value.numerator / value.denominator, digits)


def difference_record(reference: Fraction, candidate_int: int, profile: FixedProfile, tolerance: Fraction):
    candidate = Fraction(candidate_int, profile.scale)
    difference = abs(reference - candidate)
    return {
        "reference": fraction_float(reference),
        "candidate": fraction_float(candidate),
        "candidate_int": candidate_int,
        "abs_difference": fraction_float(difference, 12),
        "tolerance": fraction_float(tolerance, 12),
        "exceeded": difference > tolerance,
    }
