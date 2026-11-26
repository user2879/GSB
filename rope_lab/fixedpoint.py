"""Deterministic signed fixed-point quantization and accumulator simulation."""
from __future__ import annotations

import math
from typing import Any


def scale_for(fraction_bits: int) -> int:
    return 1 << int(fraction_bits)


def quantize_real(value: float, fraction_bits: int, quant_level: str) -> int:
    """Quantize a finite real value to a signed Python integer."""
    scaled = float(value) * scale_for(fraction_bits)
    if not math.isfinite(scaled):
        raise ValueError("cannot quantize non-finite value")

    if quant_level == "truncate":
        return math.trunc(scaled)

    if quant_level == "nearest":
        # Half away from zero; deterministic and branch-visible in reports.
        if scaled >= 0.0:
            return math.floor(scaled + 0.5)
        return math.ceil(scaled - 0.5)

    if quant_level == "ties_even":
        integer = math.floor(scaled)
        fraction = scaled - integer
        if fraction > 0.5 or (fraction == 0.5 and (integer % 2 != 0)):
            integer += 1
        return integer

    raise ValueError(f"unknown quant_level: {quant_level}")


def dequantize(integer_value: int, fraction_bits: int) -> float:
    return integer_value / scale_for(fraction_bits)


def requantize_product(product: int, fraction_bits: int, quant_level: str) -> int:
    """Convert an S*S product back to an S-scaled fixed integer."""
    unit = scale_for(fraction_bits)
    if quant_level == "truncate":
        if product >= 0:
            return product >> fraction_bits
        return -((-product) >> fraction_bits)

    if quant_level == "nearest":
        half = unit >> 1
        if product >= 0:
            return (product + half) // unit
        return -((-product + half) // unit)

    if quant_level == "ties_even":
        quotient, remainder = divmod(product, unit)
        twice_remainder = remainder * 2
        if twice_remainder > unit or (twice_remainder == unit and quotient % 2 != 0):
            quotient += 1
        return quotient

    raise ValueError(f"unknown quant_level: {quant_level}")


def signed_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def wrap_signed(value: int, bits: int) -> tuple[int, bool]:
    """Two's-complement wrap; Python arbitrary ints are used between wraps."""
    lower, upper = signed_range(bits)
    if lower <= value <= upper:
        return value, False
    span = 1 << bits
    return lower + ((value - lower) % span), True


def in_signed_range(value: int, bits: int) -> bool:
    lower, upper = signed_range(bits)
    return lower <= value <= upper


def fixed_config_view(config: dict[str, Any]) -> dict[str, Any]:
    fixed = config["fixed_point"]
    return {
        "fraction_bits": int(fixed["fraction_bits"]),
        "signed_bits": int(fixed["signed_bits"]),
        "accumulator_bits": int(fixed["accumulator_bits"]),
        "quant_level": str(fixed["quant_level"]),
    }
