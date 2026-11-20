from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, localcontext
from typing import Any, Mapping

from .canonical import decimal_text
from .model import TaskSpec


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
SEGMENTS = ("source_quantization", "formula_truncation", "aggregation_synthesis")


def trunc_divmod(numerator: int, denominator: int) -> tuple[int, int]:
    """Integer division and remainder truncated toward zero."""

    if denominator == 0:
        raise ZeroDivisionError("fixed-point denominator must not be zero")
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        quotient = -quotient
    remainder = numerator - quotient * denominator
    return quotient, remainder


def trunc_div(numerator: int, denominator: int) -> int:
    return trunc_divmod(numerator, denominator)[0]


def saturate_int64(value: int) -> int:
    return min(INT64_MAX, max(INT64_MIN, value))


def ratio_text(numerator: Decimal, denominator: Decimal) -> str:
    if denominator == 0:
        return "0.000000"
    with localcontext() as context:
        context.prec = 60
        ratio = numerator / denominator
    return format(ratio.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")


@dataclass(frozen=True)
class FixedNumber:
    value: Decimal
    fixed: int
    residual: Decimal
    saturated: bool

    @property
    def error(self) -> Decimal:
        return abs(self.residual)

    def payload(self) -> dict[str, Any]:
        return {
            "value": decimal_text(self.value),
            "fixed": self.fixed,
            "residual": decimal_text(self.residual),
            "absolute_residual": decimal_text(self.error),
            "saturated": self.saturated,
        }


class FixedQuantizer:
    """One deterministic fixed-point grid shared by values and coefficients."""

    def __init__(self, scale_power: int, overflow_policy: str = "saturate_int64") -> None:
        if overflow_policy != "saturate_int64":
            raise ValueError(f"unsupported overflow policy {overflow_policy!r}")
        self.scale_power = scale_power
        self.scale = 10**scale_power
        self.overflow_policy = overflow_policy
        self.unit = Decimal(1) / Decimal(self.scale)

    def quantize(self, value: Decimal) -> FixedNumber:
        with localcontext() as context:
            context.prec = 80
            scaled = value * self.scale
            raw = int(scaled.to_integral_value(rounding=ROUND_DOWN))
            fixed = saturate_int64(raw)
            residual = value - Decimal(fixed) / Decimal(self.scale)
        return FixedNumber(value=value, fixed=fixed, residual=residual, saturated=fixed != raw)

    def policy(self) -> dict[str, Any]:
        return {
            "scale_power": self.scale_power,
            "scale": self.scale,
            "quantization": "truncate_toward_zero",
            "arithmetic": "integer_multiply_add_then_truncate_toward_zero",
            "overflow": "saturate_signed_int64",
            "int64_min": INT64_MIN,
            "int64_max": INT64_MAX,
            "unit_error_bound": decimal_text(self.unit),
        }


def heat_level(error_bound: Decimal, unit: Decimal) -> str:
    if error_bound == 0:
        return "none"
    if error_bound <= unit:
        return "low"
    if error_bound <= unit * 10:
        return "medium"
    return "high"


def _segment(name: str, contribution: Decimal) -> dict[str, Any]:
    return {"segment": name, "contribution": contribution}


def evaluate_task(
    task: TaskSpec,
    key_results: Mapping[str, Mapping[str, Any]],
    quantizer: FixedQuantizer,
    dominance_threshold: Decimal,
) -> dict[str, Any]:
    missing = [key for key in task.keys if not key_results.get(key, {}).get("resolved")]
    if missing:
        return {
            "task_id": task.id,
            "formula": task.formula,
            "status": "unavailable",
            "missing_keys": sorted(missing),
            "scale_power": quantizer.scale_power,
            "fixed_scale": quantizer.scale,
        }

    total_num = 0
    exact_result = Decimal(0)
    source_contribution = Decimal(0)
    coefficient_contribution = Decimal(0)
    cross_contribution = Decimal(0)
    theoretical_source = Decimal(0)
    theoretical_coefficient = Decimal(0)
    theoretical_cross = Decimal(0)
    intermediates: list[dict[str, Any]] = []
    saturation = False

    for term in task.terms:
        coefficient = quantizer.quantize(term.coefficient)
        saturation = saturation or coefficient.saturated
        if term.key is None:
            constant_contribution = abs(coefficient.residual)
            constant_bound = quantizer.unit
            total_num += coefficient.fixed * quantizer.scale
            exact_result += term.coefficient
            coefficient_contribution += constant_contribution
            theoretical_coefficient += constant_bound
            intermediates.append(
                {
                    "key": None,
                    "term_kind": "constant",
                    "coefficient": decimal_text(term.coefficient),
                    "coefficient_fixed": coefficient.fixed,
                    "coefficient_residual": decimal_text(coefficient.residual),
                    "source_quantization_error": "0",
                    "coefficient_quantization_error": decimal_text(constant_contribution),
                    "cross_quantization_error": "0",
                }
            )
            continue

        key_result = key_results[term.key]
        value = Decimal(key_result["effective_raw"])
        value_fixed = int(key_result["effective_fixed"])
        value_quantization = quantizer.quantize(value)
        saturation = saturation or bool(key_result.get("saturated"))
        source_error = abs(coefficient.value) * abs(value_quantization.residual)
        coefficient_error = abs(value) * abs(coefficient.residual)
        cross_error = abs(value_quantization.residual) * abs(coefficient.residual)
        source_contribution += source_error
        coefficient_contribution += coefficient_error
        cross_contribution += cross_error
        theoretical_source += abs(coefficient.value) * quantizer.unit
        theoretical_coefficient += abs(value) * quantizer.unit
        theoretical_cross += quantizer.unit * quantizer.unit
        total_num += coefficient.fixed * value_fixed
        exact_result += term.coefficient * value
        intermediates.append(
            {
                "key": term.key,
                "term_kind": "key",
                "coefficient": decimal_text(term.coefficient),
                "coefficient_fixed": coefficient.fixed,
                "coefficient_residual": decimal_text(coefficient.residual),
                "value": decimal_text(value),
                "value_fixed": value_fixed,
                "value_residual": decimal_text(value_quantization.residual),
                "product_fixed": coefficient.fixed * value_fixed,
                "source_quantization_error": decimal_text(source_error),
                "coefficient_quantization_error": decimal_text(coefficient_error),
                "cross_quantization_error": decimal_text(cross_error),
            }
        )

    truncated_fixed, remainder = trunc_divmod(total_num, quantizer.scale)
    result_fixed = saturate_int64(truncated_fixed)
    saturation = saturation or result_fixed != truncated_fixed
    saturation_gap = Decimal(abs(result_fixed - truncated_fixed)) / Decimal(quantizer.scale)
    # total_num is scaled by scale^2 (coefficient_fixed * value_fixed).
    # The integer division only chooses an integer fixed-point result, so the
    # actual original-unit truncation error is remainder / scale^2.
    truncation_contribution = Decimal(abs(remainder)) / (Decimal(quantizer.scale) ** 2)
    computed_result = Decimal(result_fixed) / Decimal(quantizer.scale)

    formula_contribution = coefficient_contribution + truncation_contribution
    synthesis_contribution = cross_contribution + saturation_gap
    error_bound = source_contribution + formula_contribution + synthesis_contribution
    theoretical_bound = (
        theoretical_source
        + theoretical_coefficient
        + quantizer.unit
        + theoretical_cross
        + saturation_gap
    )
    actual_error = abs(exact_result - computed_result)

    contribution_map = {
        "source_quantization": source_contribution,
        "formula_truncation": formula_contribution,
        "aggregation_synthesis": synthesis_contribution,
    }
    segments: list[dict[str, Any]] = []
    ratios: list[tuple[str, Decimal]] = []
    for segment_name in SEGMENTS:
        contribution = contribution_map[segment_name]
        ratio = Decimal(0) if error_bound == 0 else contribution / error_bound
        ratios.append((segment_name, ratio))
        segments.append(
            {
                "segment": segment_name,
                "contribution": decimal_text(contribution),
                "ratio": ratio_text(contribution, error_bound),
            }
        )

    dominant_segment = "none"
    if error_bound > 0:
        best_name, best_ratio = max(ratios, key=lambda item: (item[1], -SEGMENTS.index(item[0])))
        if best_ratio >= dominance_threshold:
            dominant_segment = best_name

    return {
        "task_id": task.id,
        "formula": task.formula,
        "status": "ok",
        "scale_power": quantizer.scale_power,
        "fixed_scale": quantizer.scale,
        "referenced_keys": list(task.keys),
        "fixed_result": result_fixed,
        "result": decimal_text(computed_result),
        "exact_result": decimal_text(exact_result),
        "actual_error": decimal_text(actual_error),
        "error_bound": decimal_text(error_bound),
        "theoretical_error_bound": decimal_text(theoretical_bound),
        "bounds_valid": actual_error <= error_bound,
        "saturated": saturation,
        "saturation_gap": decimal_text(saturation_gap),
        "truncated_fixed": truncated_fixed,
        "truncation_remainder": remainder,
        "truncation_direction": "toward_zero",
        "attribution": {
            "threshold": decimal_text(dominance_threshold),
            "rule": "dominant iff contribution/error_bound >= threshold; ties follow source_quantization, formula_truncation, aggregation_synthesis",
            "dominant_segment": dominant_segment,
            "segments": segments,
        },
        "intermediates": intermediates,
    }


def compare_keys(key_results: Mapping[str, Mapping[str, Any]], quantizer: FixedQuantizer) -> list[dict[str, Any]]:
    resolved = sorted(
        ((key, result) for key, result in key_results.items() if result.get("resolved")),
        key=lambda item: item[0],
    )
    comparisons: list[dict[str, Any]] = []
    for left_index in range(len(resolved)):
        left_key, left = resolved[left_index]
        for right_index in range(left_index + 1, len(resolved)):
            right_key, right = resolved[right_index]
            left_fixed = int(left["effective_fixed"])
            right_fixed = int(right["effective_fixed"])
            difference_fixed = abs(left_fixed - right_fixed)
            difference = Decimal(difference_fixed) / Decimal(quantizer.scale)
            combined_bound = Decimal(left["comparison_error_bound"]) + Decimal(right["comparison_error_bound"])
            affected = difference <= combined_bound
            if affected:
                ordered = sorted((left_key, right_key))
                decision_basis = "error_bound_then_key_id"
            elif left_fixed < right_fixed or (left_fixed == right_fixed and left_key < right_key):
                ordered = [left_key, right_key]
                decision_basis = "fixed_value_then_key_id"
            else:
                ordered = [right_key, left_key]
                decision_basis = "fixed_value_then_key_id"
            comparisons.append(
                {
                    "key_a": left_key,
                    "key_b": right_key,
                    "fixed_a": left_fixed,
                    "fixed_b": right_fixed,
                    "difference_fixed": difference_fixed,
                    "difference": decimal_text(difference),
                    "combined_error_bound": decimal_text(combined_bound),
                    "affected_by_error_bound": affected,
                    "ordered_keys": ordered,
                    "decision_basis": decision_basis,
                }
            )
    return comparisons
