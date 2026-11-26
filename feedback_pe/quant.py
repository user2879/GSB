from __future__ import annotations

import math
from typing import Any

TIER_BITS = {"Q8": 8, "Q16": 16, "Q32": 32}


def scale_for_frac_bits(frac_bits: int) -> int:
    return 1 << int(frac_bits)


def tier_bounds(quant_tier: str) -> tuple[int, int]:
    bits = TIER_BITS[quant_tier]
    maximum = (1 << (bits - 1)) - 1
    return -(1 << (bits - 1)), maximum


def quantize_scalar(value: float, frac_bits: int, quant_tier: str) -> dict[str, Any]:
    scale = scale_for_frac_bits(frac_bits)
    min_int, max_int = tier_bounds(quant_tier)
    wide_int = int(round(float(value) * scale))
    clamped_int = max(min_int, min(max_int, wide_int))
    dequantized = clamped_int / scale
    return {
        "step": 1.0 / scale,
        "scale": scale,
        "unclamped_int": wide_int,
        "int_value": clamped_int,
        "residual": float(value) - dequantized,
        "clipped": clamped_int != wide_int,
        "dequantized": dequantized,
    }


def quantize_vector(values: list[float], frac_bits: int,
                    quant_tier: str) -> list[dict[str, Any]]:
    return [quantize_scalar(value, frac_bits, quant_tier) for value in values]


def int_vector(values: list[float], frac_bits: int, quant_tier: str) -> list[int]:
    return [item["int_value"] for item in quantize_vector(values, frac_bits, quant_tier)]


def round_shift_half_even(total: int, frac_bits: int) -> int:
    """Round a fixed-point product sum with 2*frac fractional bits to frac bits."""
    if frac_bits <= 0:
        return int(total)
    unit = 1 << int(frac_bits)
    negative = total < 0
    magnitude = abs(int(total))
    quotient, remainder = divmod(magnitude, unit)
    twice_remainder = remainder * 2
    if twice_remainder > unit or (twice_remainder == unit and quotient % 2 == 1):
        quotient += 1
    return -quotient if negative else quotient


def relative_or_absolute_error(reference: float, actual: float) -> dict[str, Any]:
    absolute_error = abs(float(actual) - float(reference))
    if reference == 0.0:
        return {
            "branch": "absolute_zero_reference",
            "reference": 0.0,
            "actual": float(actual),
            "absolute_error": absolute_error,
            "relative_error": None,
            "metric": absolute_error,
        }
    relative_error = absolute_error / abs(float(reference))
    return {
        "branch": "relative",
        "reference": float(reference),
        "actual": float(actual),
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "metric": relative_error,
    }
