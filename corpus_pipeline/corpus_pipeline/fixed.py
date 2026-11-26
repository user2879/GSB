from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from typing import Any

from .errors import Code, fail


ALLOWED_DECIMAL_PLACES = (2, 3, 4, 5, 6, 7, 8)
ALLOWED_QUANTIZATION_TIERS = (1, 10, 100, 1000)


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    original: str
    fixed: int
    residual_units: str
    scale: int
    step_units: int
    quantized: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "fixed": self.fixed,
            "residual_units": self.residual_units,
            "scale": self.scale,
            "step_units": self.step_units,
            "quantization_step": str(Decimal(self.step_units) / Decimal(self.scale)),
            "quantized": self.quantized,
        }


@dataclass(frozen=True, slots=True)
class FixedDivision:
    quotient: int
    residual_numerator: int
    numerator: int
    denominator: int
    scale: int

    @property
    def value(self) -> Decimal:
        return Decimal(self.quotient) / Decimal(self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed": self.quotient,
            "residual_numerator": self.residual_numerator,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "scale": self.scale,
            "quantization_step": str(Decimal(1) / Decimal(self.scale)),
            "decimal": str(self.value),
        }


def parse_decimal(value: Any, *, field_name: str = "value") -> Decimal:
    if isinstance(value, bool):
        fail(Code.INVALID_WEIGHT, f"{field_name} must be a number, not boolean", field=field_name)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            fail(Code.INVALID_WEIGHT, f"{field_name} is not a valid decimal", field=field_name, value=value)
    if isinstance(value, Decimal):
        return value
    fail(Code.INVALID_WEIGHT, f"{field_name} is not numeric", field=field_name, value=type(value).__name__)


def quantize_weight(
    value: Any,
    decimal_places: int,
    tier_units: int,
    *,
    allow_zero: bool = True,
    field_name: str = "weight",
) -> QuantizedWeight:
    if decimal_places not in ALLOWED_DECIMAL_PLACES:
        fail(
            Code.INVALID_PRECISION,
            "decimal_places is outside the allowed enumeration",
            value=decimal_places,
            allowed=list(ALLOWED_DECIMAL_PLACES),
        )
    if tier_units not in ALLOWED_QUANTIZATION_TIERS:
        fail(
            Code.INVALID_QUANTIZATION_TIER,
            "quantization tier is outside the allowed enumeration",
            value=tier_units,
            allowed=list(ALLOWED_QUANTIZATION_TIERS),
        )
    decimal = parse_decimal(value, field_name=field_name)
    if not decimal.is_finite():
        fail(Code.INVALID_WEIGHT, f"{field_name} must be finite", field=field_name)
    if decimal < 0:
        fail(Code.INVALID_WEIGHT, f"{field_name} must be non-negative", field=field_name, value=str(decimal))
    if not allow_zero and decimal == 0:
        fail(Code.INVALID_WEIGHT, f"{field_name} must be positive", field=field_name)

    scale = 10 ** decimal_places
    raw_units = decimal * scale
    stepped = (raw_units / Decimal(tier_units)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    fixed = int(stepped * tier_units)
    residual = raw_units - Decimal(fixed)
    return QuantizedWeight(
        original=str(decimal),
        fixed=fixed,
        residual_units=str(residual),
        scale=scale,
        step_units=tier_units,
        quantized=str(Decimal(fixed) / Decimal(scale)),
    )


def fixed_div(numerator: int, denominator: int, scale: int) -> FixedDivision:
    if denominator <= 0:
        fail(Code.INVALID_SCHEMA, "fixed division requires a positive denominator", denominator=denominator)
    if denominator < 0 or scale <= 0:
        fail(Code.INVALID_SCHEMA, "fixed division requires non-negative numerator and positive scale")
    raw = numerator * scale
    quotient, remainder = divmod(raw, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    residual = raw - quotient * denominator
    return FixedDivision(
        quotient=quotient,
        residual_numerator=residual,
        numerator=numerator,
        denominator=denominator,
        scale=scale,
    )


def similarity_fixed(intersection: int, union: int, scale: int) -> FixedDivision:
    if intersection < 0 or union < 0 or intersection > union:
        fail(
            Code.INVALID_SCHEMA,
            "invalid Jaccard components",
            intersection=intersection,
            union=union,
        )
    if union == 0:
        # Two empty subword sets are identical.  This avoids a vacuous 0/0.
        return FixedDivision(quotient=scale, residual_numerator=0, numerator=1, denominator=1, scale=scale)
    return fixed_div(intersection, union, scale)


def threshold_fixed(permille: int, scale: int) -> FixedDivision:
    return fixed_div(permille, 1000, scale)


def to_permille(fixed: int, scale: int) -> int:
    return fixed_div(fixed, scale, 1000).quotient


def rescale_fixed(value: int, source_scale: int, target_scale: int) -> int:
    if source_scale == target_scale:
        return value
    return fixed_div(value, source_scale, target_scale).quotient


def fraction_slot_key(slot: int, weight_fixed: int) -> Fraction:
    """Weighted-fair queue virtual finish time."""

    if weight_fixed <= 0:
        return Fraction(10 ** 30, 1)
    return Fraction(slot, weight_fixed)
