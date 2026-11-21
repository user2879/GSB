from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def decimal_from(value: object, *, default: str | None = None) -> Decimal:
    if value is None and default is not None:
        value = default
    if isinstance(value, bool):
        raise ValueError("boolean is not a decimal")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        return Decimal(str(value))
    raise ValueError("not a decimal-compatible value")


def round_half_up(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        return -round_half_up(-numerator, denominator)
    return (numerator * 2 + denominator) // (2 * denominator)


def quantize_ratio(
    numerator: int,
    denominator: int,
    decimal_places: int,
    quantization_step: int,
) -> int:
    if denominator <= 0:
        return 0
    scale = 10**decimal_places
    value = round_half_up(numerator * scale, denominator)
    if quantization_step > 1:
        value = ((value + quantization_step // 2) // quantization_step) * quantization_step
    return value


def format_fixed(value: int, decimal_places: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if decimal_places == 0:
        return f"{sign}{value}"
    scale = 10**decimal_places
    whole, fraction = divmod(value, scale)
    return f"{sign}{whole}.{fraction:0{decimal_places}d}"


def ratio_decimal(numerator: int, denominator: int, decimal_places: int) -> str:
    if denominator <= 0:
        return format_fixed(0, decimal_places)
    scale = 10**decimal_places
    return format_fixed(round_half_up(numerator * scale, denominator), decimal_places)


def decimal_ratio_fixed(value: Decimal, decimal_places: int) -> int:
    scale = 10**decimal_places
    return int((value * scale).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def decimal_ratio_string(value: Decimal, decimal_places: int) -> str:
    quantum = Decimal(1).scaleb(-decimal_places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), f".{decimal_places}f")
