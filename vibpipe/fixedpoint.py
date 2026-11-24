from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


GEARS: dict[str, int] = {
    "Q3": 3,
    "Q6": 6,
    "Q9": 9,
    "Q12": 12,
    "Q15": 15,
    "Q18": 18,
}
GEAR_ORDER = ["Q3", "Q6", "Q9", "Q12", "Q15", "Q18"]

LN2_DIGITS = (
    "0.693147180559945309417232121458176568075500134360255254120680009"
)


class FixedPointError(ValueError):
    pass


def gear_width(gear: str) -> int:
    try:
        return GEARS[gear]
    except KeyError as exc:
        raise FixedPointError(f"unsupported quantization gear: {gear}") from exc


def next_gear(gear: str) -> str:
    try:
        return GEAR_ORDER[GEAR_ORDER.index(gear) + 1]
    except (ValueError, IndexError) as exc:
        raise FixedPointError(f"no higher precision gear for {gear}") from exc


def decimal_to_q(value: Decimal | str | int | float, scale: int) -> int:
    if scale <= 0:
        raise FixedPointError("scale must be positive")
    if isinstance(value, float):
        value = Decimal(str(value))
    elif not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite():
        raise FixedPointError("value must be finite")
    scaled = value * Decimal(scale)
    return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))


def q_to_decimal(q: int, scale: int) -> Decimal:
    return Decimal(q) / Decimal(scale)


def q_to_json(q: int, scale: int) -> str:
    return format(q_to_decimal(q, scale), "f")


def round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise FixedPointError("denominator must be positive")
    if numerator >= 0:
        return (2 * numerator + denominator) // (2 * denominator)
    return -round_div(-numerator, denominator)


def convert_q(q: int, from_scale: int, to_scale: int) -> int:
    if from_scale == to_scale:
        return q
    return round_div(q * to_scale, from_scale)


def mul_div(a: int, b: int, scale: int) -> int:
    return round_div(a * b, scale)


def quantize_decimal_gear(q: int, scale: int, gear: str) -> int:
    """Round q to the gear's decimal grid while retaining stage scale."""

    width = gear_width(gear)
    grid = Decimal(10) ** width
    if scale <= 0:
        raise FixedPointError("scale must be positive")
    grid_q = Decimal(scale) / grid
    if grid_q == grid_q.to_integral_value():
        step = int(grid_q)
        if step <= 1:
            return q
        units = round_div(q, step)
        return units * step
    # Exact rational path for unusual stage scales.
    quantized = (q_to_decimal(q, scale) * grid).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return decimal_to_q(Decimal(int(quantized)) / grid, scale)


def half_step_q(scale: int, gear: str) -> Decimal:
    width = gear_width(gear)
    return Decimal(scale) / (Decimal(2) * (Decimal(10) ** width))


def fixed_ln(q: int, scale: int, iterations: int = 12) -> int:
    """Natural logarithm in integer fixed point at the supplied scale."""

    if q <= 0:
        raise FixedPointError("log argument must be positive")
    if iterations < 1:
        raise FixedPointError("log iterations must be positive")

    value_q = q
    exponent = 0
    one = scale
    two = 2 * scale

    while value_q >= two:
        value_q = round_div(value_q, 2)
        exponent += 1
    while value_q < one:
        value_q = value_q * 2
        exponent -= 1

    ratio_q = round_div((value_q - scale) * scale, value_q + scale)
    power_q = ratio_q
    ln_mantissa_q = 0
    for index in range(iterations):
        odd = 2 * index + 1
        term_q = round_div(power_q, odd)
        ln_mantissa_q += 2 * term_q
        power_q = round_div(power_q * ratio_q * ratio_q, scale * scale)

    ln2_q = decimal_to_q(Decimal(LN2_DIGITS), scale)
    return exponent * ln2_q + ln_mantissa_q


def as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedPointError(f"{name} must be an integer")
    return value


def finite_decimal(value: Any) -> bool:
    try:
        return Decimal(str(value)).is_finite()
    except (InvalidOperation, ValueError):
        return False
