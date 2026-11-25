from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from .config import FixedConfig
from .model import FIELD_RANK


STAGE_ORDER = (
    "inverse_document_frequency",
    "field_length_normalization",
    "term_frequency_saturation",
    "field_weight",
    "accumulation",
)


def quantize_fraction(value: Fraction, quantum: int) -> int:
    """Round a rational value to the nearest multiple of ``quantum``.

    Ties are resolved away from zero.  The input/output units are fixed-point
    least-significant-place units, not decimal real numbers.
    """

    if quantum <= 0:
        raise ValueError("quantum must be positive")
    if value == 0:
        return 0
    sign = -1 if value < 0 else 1
    numerator = abs(value.numerator)
    denominator = abs(value.denominator) * quantum
    rounded_units = (2 * numerator + denominator) // (2 * denominator)
    return sign * rounded_units * quantum


def quantize_decimal(value: Decimal, quantum: int) -> int:
    return quantize_fraction(Fraction(value), quantum)


def quantize_float(value: float, quantum: int) -> int:
    return quantize_fraction(Fraction(Decimal(str(value))), quantum)


def fixed_from_value(value: int | float, fixed: FixedConfig) -> int:
    scale = fixed.scale
    if isinstance(value, int) and not isinstance(value, bool):
        numerator: int | Fraction = value * scale
    else:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("fixed value must be finite")
        numerator = Fraction(Decimal(str(value))) * scale
    return quantize_fraction(Fraction(numerator), fixed.quantum)


def fixed_units_to_float(units: int, fixed: FixedConfig) -> float:
    return units / fixed.scale


def fixed_mul(left: int, right: int, fixed: FixedConfig) -> int:
    return quantize_fraction(Fraction(left * right, fixed.scale), fixed.quantum)


def fixed_div(numerator: int, denominator: int, fixed: FixedConfig) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    return quantize_fraction(
        Fraction(numerator * fixed.scale, denominator), fixed.quantum
    )


def fixed_average(total: int, count: int, fixed: FixedConfig) -> int:
    if count <= 0:
        return 0
    return quantize_fraction(
        Fraction(total * fixed.scale, count), fixed.quantum
    )


def fixed_idf(document_count: int, document_frequency: int, fixed: FixedConfig) -> int:
    if document_count <= 0 or document_frequency <= 0:
        return 0
    with localcontext() as context:
        context.prec = 60
        ratio = Decimal(1) + (
            Decimal(document_count - document_frequency) + Decimal("0.5")
        ) / (Decimal(document_frequency) + Decimal("0.5"))
        value = ratio.ln() * Decimal(fixed.scale)
        return quantize_decimal(value, fixed.quantum)


def float_idf(document_count: int, document_frequency: int) -> float:
    if document_count <= 0 or document_frequency <= 0:
        return 0.0
    return math.log(
        1.0
        + (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def saturation_fixed(
    tf: int,
    field_length: int,
    average_field_length: int,
    k1: int,
    b: int,
    fixed: FixedConfig,
) -> dict[str, int]:
    scale = fixed.scale
    if tf <= 0 or average_field_length <= 0 or field_length < 0:
        return {
            "length_ratio": 0,
            "length_norm": 0,
            "denominator_base": 0,
            "denominator": 0,
            "saturation": 0,
        }
    length_ratio = quantize_fraction(
        Fraction(field_length * scale * scale, average_field_length),
        fixed.quantum,
    )
    length_norm = fixed_mul(b, length_ratio, fixed)
    denominator_base = scale - b + length_norm
    denominator = tf * scale + fixed_mul(k1, denominator_base, fixed)
    numerator = tf * (k1 + scale)
    saturation = fixed_div(numerator, denominator, fixed)
    return {
        "length_ratio": length_ratio,
        "length_norm": length_norm,
        "denominator_base": denominator_base,
        "denominator": denominator,
        "saturation": saturation,
    }


def saturation_float(
    tf: int,
    field_length: int,
    average_field_length: float,
    k1: float,
    b: float,
) -> dict[str, float]:
    if tf <= 0 or average_field_length <= 0:
        return {
            "length_ratio": 0.0,
            "length_norm": 0.0,
            "denominator_base": 0.0,
            "denominator": 0.0,
            "saturation": 0.0,
        }
    length_ratio = field_length / average_field_length
    length_norm = b * length_ratio
    denominator_base = 1.0 - b + length_norm
    denominator = tf + k1 * denominator_base
    saturation = tf * (k1 + 1.0) / denominator
    return {
        "length_ratio": length_ratio,
        "length_norm": length_norm,
        "denominator_base": denominator_base,
        "denominator": denominator,
        "saturation": saturation,
    }


def stage_errors(
    field_trace: dict[str, Any], fixed: FixedConfig
) -> dict[str, float]:
    scale = fixed.scale
    return {
        "inverse_document_frequency": abs(
            field_trace["idf_fixed_units"] - field_trace["idf_float"] * scale
        ),
        "field_length_normalization": abs(
            field_trace["length_norm_fixed_units"]
            - field_trace["length_norm_float"] * scale
        ),
        "term_frequency_saturation": abs(
            field_trace["saturation_fixed_units"]
            - field_trace["saturation_float"] * scale
        ),
        "field_weight": abs(
            field_trace["field_weight_fixed_units"]
            - field_trace["field_weight_float"] * scale
        ),
        "accumulation": abs(
            field_trace["contribution_fixed_units"]
            - field_trace["contribution_float"] * scale
        ),
    }


def first_tolerance_stage(
    errors: dict[str, float], tolerance_units: float
) -> str:
    for stage in STAGE_ORDER:
        if errors.get(stage, 0.0) > tolerance_units:
            return stage
    return "none"


def sort_field_names(fields: tuple[str, ...] | list[str]) -> list[str]:
    return sorted(fields, key=lambda name: FIELD_RANK[name])
