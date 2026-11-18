from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from .canonical import decimal_text


def quantum_fixed(quantum: Decimal, decimal_places: int) -> int:
    with localcontext() as context:
        context.prec = 80
        scaled = quantum.scaleb(decimal_places)
        integral = scaled.to_integral_value()
        if scaled != integral:
            raise ValueError("quantum is not representable at the requested precision")
        return int(integral)


def decimal_to_fixed(value: Decimal, decimal_places: int, quantum: Decimal) -> int:
    q_fixed = quantum_fixed(quantum, decimal_places)
    with localcontext() as context:
        context.prec = 80
        scaled = value.scaleb(decimal_places)
        units = (scaled / Decimal(q_fixed)).to_integral_value(rounding=ROUND_HALF_EVEN)
        return int(units) * q_fixed


def round_fixed_to_quantum(
    value_fixed: int, decimal_places: int, quantum: Decimal
) -> int:
    q_fixed = quantum_fixed(quantum, decimal_places)
    with localcontext() as context:
        context.prec = 80
        units = (Decimal(value_fixed) / Decimal(q_fixed)).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
        return int(units) * q_fixed


def convert_fixed(value_fixed: int, from_places: int, to_places: int) -> int:
    if from_places == to_places:
        return value_fixed
    with localcontext() as context:
        context.prec = 80
        scaled = Decimal(value_fixed).scaleb(to_places - from_places)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def fixed_decimal(value_fixed: int, decimal_places: int) -> Decimal:
    return Decimal(value_fixed).scaleb(-decimal_places)


def fixed_text(value_fixed: int, decimal_places: int) -> str:
    return decimal_text(fixed_decimal(value_fixed, decimal_places))


def quantize_decimal(value: Decimal, decimal_places: int, quantum: Decimal) -> int:
    return decimal_to_fixed(value, decimal_places, quantum)
