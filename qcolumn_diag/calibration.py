"""Per-column quantile calibration and affine quantization parameters."""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from .models import TensorConfig


def round_half_away_from_zero_scalar(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("cannot round a non-finite value")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def round_half_away_from_zero_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def linear_quantile(sorted_values: np.ndarray, permille: int) -> float:
    """Linear interpolation on a sorted float64 vector, q = permille/1000."""

    count = int(sorted_values.size)
    if count == 0:
        raise ValueError("quantile of an empty vector")
    if count == 1:
        return float(sorted_values[0])
    position = (float(permille) / 1000.0) * float(count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, count - 1)
    fraction = position - float(lower)
    lower_value = float(sorted_values[lower])
    upper_value = float(sorted_values[upper])
    if fraction == 0.0:
        return lower_value
    return float(lower_value + fraction * (upper_value - lower_value))


def quantile_stage_artifact(
    values: np.ndarray,
    missing_mask: np.ndarray,
    tensor: TensorConfig,
    *,
    interpolation: str,
    tie_break: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (artifact, failure) for the quantile stage of one column."""

    non_missing = np.asarray(values[~missing_mask], dtype=np.float64)
    column_count = non_missing.size
    if column_count == 0:
        return None, {
            "error_code": "E_COLUMN_EMPTY",
            "field": "column.elements",
            "message": "column has no non-missing elements",
        }
    if not np.all(np.isfinite(non_missing)):
        return None, {
            "error_code": "E_SAMPLE_INVALID",
            "field": "column.elements",
            "message": "column contains non-finite non-missing elements",
        }
    order = np.argsort(non_missing, kind="stable")
    sorted_values = non_missing[order]
    lower = linear_quantile(sorted_values, tensor.quantile_lower_permille)
    upper = linear_quantile(sorted_values, tensor.quantile_upper_permille)
    first_index = int(order[0]) if order.size else None
    last_index = int(order[-1]) if order.size else None
    artifact = {
        "non_missing_count": int(column_count),
        "missing_count": int(np.count_nonzero(missing_mask)),
        "observed_min": float(sorted_values[0]),
        "observed_max": float(sorted_values[-1]),
        "lower_quantile": lower,
        "upper_quantile": upper,
        "quantile_lower_permille": int(tensor.quantile_lower_permille),
        "quantile_upper_permille": int(tensor.quantile_upper_permille),
        "interpolation": interpolation,
        "tie_break": tie_break,
        "sort_order": "ascending_stable",
        "first_sorted_source_row": first_index,
        "last_sorted_source_row": last_index,
        "all_values_equal": bool(sorted_values[0] == sorted_values[-1]),
    }
    return artifact, None


def _code_limits(bit_width: int, scheme: str) -> tuple[int, int]:
    if scheme == "symmetric":
        positive = (1 << (bit_width - 1)) - 1
        return -positive, positive
    return 0, (1 << bit_width) - 1


def quantize_scale(ideal_scale: float, fractional_bits: int) -> tuple[float, float, int]:
    """Quantize scale to multiples of 2^-fractional_bits.

    Returns (quantized_scale, scale_unit, rounded_units).
    """

    if not math.isfinite(ideal_scale) or ideal_scale < 0:
        raise ValueError("ideal scale must be a finite non-negative number")
    unit = math.ldexp(1.0, -int(fractional_bits))
    if ideal_scale == 0.0:
        return 0.0, unit, 0
    units = round_half_away_from_zero_scalar(ideal_scale / unit)
    if units <= 0:
        units = 1
    return float(units) * unit, unit, int(units)


def derive_quant_params(
    *,
    lower_quantile: float,
    upper_quantile: float,
    tensor: TensorConfig,
    bit_width: int,
) -> dict[str, Any]:
    scheme = tensor.scheme
    code_min, code_max = _code_limits(bit_width, scheme)
    if scheme == "symmetric":
        maximum_absolute = max(abs(float(lower_quantile)), abs(float(upper_quantile)))
        ideal_scale = (
            maximum_absolute / float(code_max) if code_max > 0 else 0.0
        )
    else:
        ideal_scale = (float(upper_quantile) - float(lower_quantile)) / float(
            code_max - code_min
        )
    quantized_scale, scale_unit, scale_units = quantize_scale(
        ideal_scale, tensor.fractional_bits
    )
    scale_quantization_delta = abs(quantized_scale - ideal_scale)
    scale_quantization_relative_delta = (
        scale_quantization_delta / ideal_scale if ideal_scale > 0.0 else 0.0
    )

    zero_scale = ideal_scale <= 0.0 or quantized_scale <= 0.0
    degraded_scale = False
    rejected_zero_scale = False
    if zero_scale:
        if tensor.zero_scale_policy == "clamp_to_min_scale":
            quantized_scale = scale_unit
            scale_units = 1
            degraded_scale = True
        else:
            quantized_scale = 0.0
            scale_units = 0
            rejected_zero_scale = True

    ideal_zero_point: float | int = 0.0
    zero_point = 0
    if scheme == "asymmetric":
        if ideal_scale > 0.0:
            ideal_zero_point = (
                float(code_min) - float(lower_quantile) / ideal_scale
            )
        else:
            ideal_zero_point = float(code_min)
        if quantized_scale > 0.0:
            zero_point_float = (
                float(code_min) - float(lower_quantile) / quantized_scale
            )
            zero_point = round_half_away_from_zero_scalar(zero_point_float)
            zero_point = max(code_min, min(code_max, zero_point))
        else:
            zero_point = 0
    zero_point_delta = (
        abs(float(zero_point) - float(ideal_zero_point))
        if scheme == "asymmetric"
        else 0.0
    )

    return {
        "scheme": scheme,
        "bit_width": int(bit_width),
        "quantization_levels": int(1 << bit_width),
        "code_min": int(code_min),
        "code_max": int(code_max),
        "fractional_bits": int(tensor.fractional_bits),
        "scale_unit": float(scale_unit),
        "scale_units": int(scale_units),
        "scale_quantization": "round_half_away_from_zero",
        "ideal_scale": float(ideal_scale),
        "quantized_scale": float(quantized_scale),
        "scale_quantization_delta": float(scale_quantization_delta),
        "scale_quantization_relative_delta": float(scale_quantization_relative_delta),
        "zero_point": int(zero_point),
        "ideal_zero_point": float(ideal_zero_point),
        "zero_point_quantization_delta": float(zero_point_delta),
        "zero_scale": bool(zero_scale),
        "degraded_scale": bool(degraded_scale),
        "zero_scale_policy": tensor.zero_scale_policy,
        "rejected_zero_scale": bool(rejected_zero_scale),
    }


def full_range_quant_params(
    *,
    observed_min: float,
    observed_max: float,
    tensor: TensorConfig,
    bit_width: int,
) -> dict[str, Any]:
    """Alternative parameters used only by the error-attribution counterfactual."""

    return derive_quant_params(
        lower_quantile=float(observed_min),
        upper_quantile=float(observed_max),
        tensor=tensor,
        bit_width=bit_width,
    )


def python_version() -> str:
    return sys.version.split()[0]
