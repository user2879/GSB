from __future__ import annotations

from fractions import Fraction
from typing import Any


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def format_fixed(fixed_int: int, digits: int) -> str:
    if digits <= 0:
        return str(fixed_int)
    whole = fixed_int // (10**digits)
    fraction = fixed_int % (10**digits)
    return f"{whole}.{fraction:0{digits}d}"


def quantize_score(score: Fraction, quant_config: dict[str, Any]) -> dict[str, Any]:
    digits = int(quant_config.get("frac_digits", 3))
    level = str(quant_config.get("level", "standard"))
    multipliers = quant_config.get(
        "level_multipliers", {"fine": 1, "standard": 5, "coarse": 25}
    )
    quantum = int(quant_config.get("quantum_multiplier", multipliers.get(level, 5)))
    scale = 10**digits
    scaled_exact = score * scale
    fixed_int = floor_fraction(scaled_exact)
    residual = scaled_exact - fixed_int

    quantum_index = floor_fraction(scaled_exact / quantum)
    base_int = quantum_index * quantum
    quant_residual = scaled_exact - base_int
    twice_residual = quant_residual * 2
    if quant_residual == 0:
        direction = "exact"
        quantized_int = base_int
    elif twice_residual >= quantum:
        quantized_int = base_int + quantum
        direction = "up_tie" if twice_residual == quantum else "up"
    else:
        quantized_int = base_int
        direction = "down"

    high_digits = digits + 1
    high_scale = 10**high_digits
    high_exact = score * high_scale
    high_floor = floor_fraction(high_exact)
    high_fraction = high_exact - high_floor
    high_reference_int = high_floor + (1 if high_fraction * 2 >= 1 else 0)

    return {
        "frac_digits": digits,
        "level": level,
        "scale": scale,
        "quantum": quantum,
        "quantum_step_value": format_fixed(quantum, digits),
        "half_step_value": format_fixed_fraction(Fraction(quantum, 2 * scale), digits + 2),
        "unquantized_fixed_int": fixed_int,
        "fixed_residual_numerator": residual.numerator,
        "fixed_residual_denominator": residual.denominator,
        "quantum_base_int": base_int,
        "quantized_fixed_int": quantized_int,
        "quantized_value": format_fixed(quantized_int, digits),
        "rounding_direction": direction,
        "quantization_residual_numerator": quant_residual.numerator,
        "quantization_residual_denominator": quant_residual.denominator,
        "high_width_digits": high_digits,
        "high_width_reference_int": high_reference_int,
        "high_width_reference_value": format_fixed(high_reference_int, high_digits),
    }


def format_fixed_fraction(value: Fraction, digits: int) -> str:
    scale = 10**digits
    rounded = floor_fraction(value * scale * 2 + 1) // 2
    return format_fixed(rounded, digits)


def boundary_information(
    score: Fraction, threshold_permille: int, quantized: dict[str, Any]
) -> dict[str, Any]:
    threshold = Fraction(int(threshold_permille), 1000)
    delta = score - threshold
    distance = abs(delta)
    half_step = Fraction(int(quantized["quantum"]), 2 * int(quantized["scale"]))
    is_boundary = distance <= half_step
    return {
        "threshold_permille": threshold_permille,
        "threshold_value": f"{threshold.numerator}/{threshold.denominator}",
        "exact_delta_numerator": delta.numerator,
        "exact_delta_denominator": delta.denominator,
        "distance_numerator": distance.numerator,
        "distance_denominator": distance.denominator,
        "half_quantum_numerator": half_step.numerator,
        "half_quantum_denominator": half_step.denominator,
        "boundary": is_boundary,
        "high_width_reference_value": quantized["high_width_reference_value"],
        "high_width_digits": quantized["high_width_digits"],
    }
