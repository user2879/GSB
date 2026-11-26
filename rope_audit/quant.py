from __future__ import annotations

import math
from typing import Any

from .errors import AuditError


def qscale(quant: dict[str, Any]) -> int:
    return 1 << int(quant["effective_frac_bits"])


def quantize_scalar(value: float, scale: int, context: str | None = None) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise AuditError(code="E_NONFINITE_INPUT", message=f"non-finite value while quantizing {context or 'value'}")
    scaled = float(value) * scale
    if scaled >= 0.0:
        return int(math.floor(scaled + 0.5))
    return -int(math.floor(-scaled + 0.5))


def dequantize_int(value: int, scale: int) -> float:
    return float(value) / float(scale)


def rounded_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise AuditError(code="E_INTEGER_KERNEL", message="rounded_div requires a positive denominator")
    negative = numerator < 0
    absolute = abs(int(numerator))
    quotient, remainder = divmod(absolute, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return -quotient if negative else quotient


def quantize_with_trace(values: list[float], quant: dict[str, Any], context: str) -> tuple[list[int], list[dict[str, Any]]]:
    scale = qscale(quant)
    integers: list[int] = []
    trace: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        int_value = quantize_scalar(value, scale, f"{context}[{index}]")
        reconstructed = dequantize_int(int_value, scale)
        integers.append(int_value)
        trace.append(
            {
                "index": index,
                "raw_value": float(value),
                "quant_step": 1.0 / scale,
                "scale": scale,
                "integer": int_value,
                "reconstructed": reconstructed,
                "residual": float(value) - reconstructed,
                "effective_frac_bits": int(quant["effective_frac_bits"]),
                "quant_level": int(quant["quant_level"]),
            }
        )
    return integers, trace


def fixed_sin_cos(angle_int: int, frac_bits: int) -> tuple[int, int]:
    """Deterministic integer sin/cos using quadrant reduction and Taylor series."""
    scale = 1 << int(frac_bits)
    pi_int = quantize_scalar(math.pi, scale, "pi")
    half_pi = quantize_scalar(math.pi / 2.0, scale, "half_pi")
    quarter_pi = quantize_scalar(math.pi / 4.0, scale, "quarter_pi")
    two_pi = 2 * pi_int

    reduced = int(angle_int) % two_pi
    quadrant = (reduced + quarter_pi) // half_pi
    y_int = reduced - quadrant * half_pi
    y_squared = rounded_div(y_int * y_int, scale)

    sin_term = y_int
    sin_sum = y_int
    for k in range(12):
        factor = (2 * k + 2) * (2 * k + 3)
        next_term = -rounded_div(sin_term * y_squared, scale * factor)
        if next_term == 0:
            break
        sin_term = next_term
        sin_sum += sin_term

    cos_term = scale
    cos_sum = scale
    for k in range(12):
        factor = (2 * k + 1) * (2 * k + 2)
        next_term = -rounded_div(cos_term * y_squared, scale * factor)
        if next_term == 0:
            break
        cos_term = next_term
        cos_sum += cos_term

    quadrant_mod = int(quadrant) % 4
    if quadrant_mod == 0:
        sin_int, cos_int = sin_sum, cos_sum
    elif quadrant_mod == 1:
        sin_int, cos_int = cos_sum, -sin_sum
    elif quadrant_mod == 2:
        sin_int, cos_int = -sin_sum, -cos_sum
    else:
        sin_int, cos_int = -cos_sum, sin_sum

    bound = scale + max(1, scale >> frac_bits)
    sin_int = max(-bound, min(bound, sin_int))
    cos_int = max(-bound, min(bound, cos_int))
    return int(sin_int), int(cos_int)


def fixed_exp_neg(z_int: int, scale: int, max_terms: int = 28) -> int:
    """Return exp(-z) * scale for non-negative fixed-point z."""
    if z_int < 0:
        raise AuditError(code="E_INTEGER_KERNEL", message="stable exp range reduction requires z_int >= 0")
    ln2_int = quantize_scalar(math.log(2.0), scale, "ln2")
    power_of_two, remainder = divmod(int(z_int), ln2_int)
    term = scale
    total = scale
    for n in range(1, max_terms + 1):
        term = -rounded_div(term * remainder, scale * n)
        total += term
        if term == 0:
            break
    if power_of_two >= (scale.bit_length() + 40):
        return 0
    return max(0, rounded_div(total, 1 << power_of_two))
