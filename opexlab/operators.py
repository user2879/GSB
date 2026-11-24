"""Deterministic integer Softmax and LayerNorm row operators."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .fixed import FixedContext, ascending_sum, exp_negative_fixed, isqrt_fixed
from .models import WeightRow
from .util import div_round_half_away, quantize_int_to_step, round_half_away


SOFTMAX_ACCUMULATION_ORDER = "column_ascending"
SOFTMAX_MAX_TIE_RULE = "first_smallest_column_index"
SOFTMAX_ZERO_DENOMINATOR_RULE = "uniform_with_remainder_at_column_zero"
LAYERNORM_ACCUMULATION_ORDER = "column_ascending"
LAYERNORM_LOW_VARIANCE_RULE = "replace_variance_with_configured_floor"


def softmax_row(
    values: Sequence[float],
    context: FixedContext,
    *,
    polynomial_order: int,
    approximation_name: str,
) -> Dict[str, Any]:
    if not values:
        raise ValueError("softmax requires a non-empty row")
    fixed_input = [context.quantize_float(value) for value in values]
    row_max = fixed_input[0]
    row_max_index = 0
    for index in range(1, len(fixed_input)):
        if fixed_input[index] > row_max:
            row_max = fixed_input[index]
            row_max_index = index
    tie_count = sum(1 for value in fixed_input if value == row_max)
    shifted = [row_max - value for value in fixed_input]
    exponentials = []
    for shifted_value in shifted:
        value = exp_negative_fixed(shifted_value, context, polynomial_order)
        exponentials.append(context.quantize_raw(value))
    denominator = ascending_sum(exponentials)
    denominator_zero = denominator == 0
    if denominator_zero:
        width = len(fixed_input)
        base = context.scale // width
        output = [base] * width
        output[0] += context.scale - sum(output)
        denominator_rule = SOFTMAX_ZERO_DENOMINATOR_RULE
    else:
        output = [
            quantize_int_to_step(
                div_round_half_away(value * context.scale, denominator),
                context.quantization_step,
            )
            for value in exponentials
        ]
        residual = context.scale - sum(output)
        output[row_max_index] += residual
        denominator_rule = "normalized_ascending_sum"
    if sum(output) != context.scale:
        # This branch is unreachable for integer arithmetic unless a future
        # strategy violates normalization.  Keep a deterministic repair path.
        output[row_max_index] += context.scale - sum(output)
    return {
        "fixed_input": fixed_input,
        "row_max_fixed": row_max,
        "row_max_index": row_max_index,
        "row_max_tie_count": tie_count,
        "shifted_fixed": shifted,
        "exp_fixed": exponentials,
        "denominator_fixed": denominator,
        "output_fixed": output,
        "output_float": [context.to_float(value) for value in output],
        "metadata": {
            "accumulation_order": SOFTMAX_ACCUMULATION_ORDER,
            "exp_approximation": approximation_name,
            "exp_polynomial_order": polynomial_order,
            "fixed_fractional_bits": context.fractional_bits,
            "quant_level": context.quant_level,
            "quantization_step_raw": context.quantization_step,
            "max_subtraction": True,
            "max_tie_rule": SOFTMAX_MAX_TIE_RULE,
            "zero_denominator_rule": denominator_rule,
            "normalization_repair_index": row_max_index,
        },
    }


def layernorm_statistics(
    values: Sequence[float],
    context: FixedContext,
    *,
    variance_floor: float,
) -> Dict[str, Any]:
    if not values:
        raise ValueError("layernorm requires a non-empty row")
    fixed_input = [context.quantize_float(value) for value in values]
    dim = len(fixed_input)
    total = ascending_sum(fixed_input)
    mean_fixed = div_round_half_away(total, dim)
    differences = [value - mean_fixed for value in fixed_input]
    squared = [difference * difference for difference in differences]
    squared_sum = ascending_sum(squared)
    variance_fixed = div_round_half_away(squared_sum, dim)
    variance_floor_fixed = max(0, round_half_away(variance_floor * context.scale * context.scale))
    floor_applied = variance_fixed < variance_floor_fixed
    if floor_applied:
        variance_used = max(1, variance_floor_fixed)
        floor_rule = LAYERNORM_LOW_VARIANCE_RULE
    elif variance_fixed == 0:
        variance_used = 1
        floor_applied = True
        floor_rule = "zero_variance_guard_one_raw_unit"
    else:
        variance_used = variance_fixed
        floor_rule = "not_applied"
    std_fixed = max(1, isqrt_fixed(variance_used))
    return {
        "fixed_input": fixed_input,
        "sum_fixed": total,
        "mean_fixed": mean_fixed,
        "differences_fixed": differences,
        "squared_differences_fixed": squared,
        "squared_sum_fixed": squared_sum,
        "variance_fixed": variance_fixed,
        "variance_floor_fixed": variance_floor_fixed,
        "variance_used_fixed": variance_used,
        "variance_floor_applied": bool(floor_applied),
        "variance_floor_rule": floor_rule,
        "std_fixed": std_fixed,
        "stats": {
            "accumulation_order": LAYERNORM_ACCUMULATION_ORDER,
            "fixed_fractional_bits": context.fractional_bits,
            "quant_level": context.quant_level,
            "quantization_step_raw": context.quantization_step,
            "sum_fixed": total,
            "mean_fixed": mean_fixed,
            "squared_sum_fixed": squared_sum,
            "variance_fixed": variance_fixed,
            "variance_floor_fixed": variance_floor_fixed,
            "variance_used_fixed": variance_used,
            "std_fixed": std_fixed,
            "variance_floor_applied": bool(floor_applied),
            "variance_floor_rule": floor_rule,
        },
    }


def layernorm_apply_from_stats(
    statistics: Dict[str, Any],
    weight: WeightRow,
    context: FixedContext,
    *,
    path_version: str,
    bias_enabled: bool,
) -> Dict[str, Any]:
    differences = statistics["differences_fixed"]
    std_fixed = statistics["std_fixed"]
    normalized = [
        context.quantize_raw(div_round_half_away(difference * context.scale, std_fixed))
        for difference in differences
    ]
    scale_fixed = context.quantize_float(weight.scale)
    if path_version == "v1":
        bias_source = int(weight.raw.get("bias_code", round_half_away(weight.bias * 1000.0)))
        bias_value = bias_source / 1000.0
        bias_enabled_effective = True
    else:
        bias_value = weight.bias if bias_enabled else 0.0
        bias_enabled_effective = bool(bias_enabled)
    bias_fixed = context.quantize_float(bias_value)
    scaled = [context.quantize_raw(context.mul(value, scale_fixed)) for value in normalized]
    output = [context.quantize_raw(value + bias_fixed) for value in scaled]
    return {
        "normalized_fixed": normalized,
        "scale_fixed": scale_fixed,
        "bias_fixed": bias_fixed,
        "bias_enabled": bias_enabled_effective,
        "output_fixed": output,
        "output_float": [context.to_float(value) for value in output],
        "stats": dict(statistics["stats"], path_version=path_version),
        "parameters_fixed": {
            "scale_fixed": scale_fixed,
            "bias_fixed": bias_fixed,
            "bias_enabled": bias_enabled_effective,
        },
    }


def layernorm_row(
    values: Sequence[float],
    weight: WeightRow,
    context: FixedContext,
    *,
    variance_floor: float,
    path_version: str,
    bias_enabled: bool,
) -> Dict[str, Any]:
    statistics = layernorm_statistics(values, context, variance_floor=variance_floor)
    applied = layernorm_apply_from_stats(
        statistics,
        weight,
        context,
        path_version=path_version,
        bias_enabled=bias_enabled,
    )
    result = dict(statistics)
    result.update(applied)
    return result
