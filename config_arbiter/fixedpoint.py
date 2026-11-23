from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from .util import ceil_fraction, decimal_text, fraction_json, trunc_div_toward_zero

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


def saturate_i64(value: int) -> int:
    if value < I64_MIN:
        return I64_MIN
    if value > I64_MAX:
        return I64_MAX
    return value


def add_i64(left: int, right: int) -> int:
    return saturate_i64(left + right)


def sub_i64(left: int, right: int) -> int:
    return saturate_i64(left - right)


def mul_i64(left: int, right: int) -> int:
    return saturate_i64(left * right)


def div_i64_trunc(left: int, right: int) -> int:
    return saturate_i64(trunc_div_toward_zero(left, right))


@dataclass(frozen=True)
class QuantizedValue:
    value_text: str
    fixed: int
    error: Fraction
    error_upper_fixed_units: int
    saturated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "value_text": self.value_text,
            "fixed": self.fixed,
            "quantization_error_fixed_units": fraction_json(self.error),
            "quantization_error_upper_fixed_units": self.error_upper_fixed_units,
            "saturated": self.saturated,
        }


def scale_factor(power: int) -> int:
    return 10**power


def quantize(value: Decimal, power: int) -> QuantizedValue:
    """Truncate toward zero after scaling, then saturate to signed int64."""
    numerator, denominator = value.as_integer_ratio()
    scaled_numerator = numerator * scale_factor(power)
    truncated = trunc_div_toward_zero(scaled_numerator, denominator)
    fixed = saturate_i64(truncated)
    saturated = fixed != truncated
    exact_scaled = Fraction(scaled_numerator, denominator)
    error = abs(exact_scaled - Fraction(fixed, 1))
    return QuantizedValue(
        value_text=decimal_text(value),
        fixed=fixed,
        error=error,
        error_upper_fixed_units=ceil_fraction(error),
        saturated=saturated,
    )


def quantize_weight(weight: int, power: int) -> int:
    return saturate_i64(weight * scale_factor(power))


def tolerance_fixed(tolerance: int, power: int) -> int:
    return saturate_i64(tolerance * scale_factor(power))


def comparison_error_bound(tolerance: int, power: int, max_error_upper: int) -> int:
    return saturate_i64(tolerance_fixed(tolerance, power) + 2 * max_error_upper)


def midpoint_trunc(left: int, right: int) -> int:
    return saturate_i64(trunc_div_toward_zero(left + right, 2))
