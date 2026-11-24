from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from .canonical import normalize_decimal

SATURATION_LIMIT = (1 << 63) - 1


@dataclass(frozen=True)
class QuantizedDecimal:
    text: str
    value: Decimal
    units: int
    error_units: Decimal


def scale_factor(exponent: int) -> int:
    return 10**exponent


def quantize_decimal(value: object, scale_exponent: int, field: str) -> QuantizedDecimal:
    text, decimal_value = normalize_decimal(value, field)
    scaled = decimal_value * scale_factor(scale_exponent)
    units = saturate(int(scaled))  # int() truncates toward zero for Decimal.
    error = abs(scaled - Decimal(units))
    return QuantizedDecimal(text=text, value=decimal_value, units=units, error_units=error)


def saturate(value: int) -> int:
    if value > SATURATION_LIMIT:
        return SATURATION_LIMIT
    if value < -SATURATION_LIMIT:
        return -SATURATION_LIMIT
    return value


def sat_add(left: int, right: int) -> int:
    return saturate(int(left) + int(right))


def sat_mul(left: int, right: int) -> int:
    return saturate(int(left) * int(right))


def trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    negative = (numerator < 0) != (denominator < 0)
    quotient = abs(int(numerator)) // abs(int(denominator))
    return -quotient if negative else quotient


def sat_mul_div(left: int, right: int, denominator: int) -> int:
    return saturate(trunc_div(sat_mul(left, right), denominator))


def safe_weight_sum(weights: Iterable[int]) -> int:
    total = 0
    for weight in weights:
        total = sat_add(total, weight)
    return total


def weighted_fixed_mean(pairs: Iterable[tuple[int, int]]) -> int:
    numerator = 0
    denominator = 0
    for weight, value in pairs:
        numerator = sat_add(numerator, sat_mul(weight, value))
        denominator = sat_add(denominator, weight)
    if denominator == 0:
        return 0
    return saturate(trunc_div(numerator, denominator))


def midpoint_fixed(low: int, high: int) -> int:
    return saturate(trunc_div(sat_add(low, high), 2))


def fixed_to_decimal_text(units: int, scale_exponent: int) -> str:
    value = Decimal(int(units)).scaleb(-scale_exponent)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def ceil_units(value: Decimal) -> int:
    if value <= 0:
        return 0
    return int(value.to_integral_value(rounding="ROUND_CEILING"))
