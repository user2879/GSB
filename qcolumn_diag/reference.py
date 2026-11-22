"""Independent float64 reference path used for acceptance checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .calibration import quantize_scale
from .models import TensorConfig


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _reference_params(
    lower: float,
    upper: float,
    tensor: TensorConfig,
    bit_width: int,
) -> dict[str, Any]:
    if tensor.scheme == "symmetric":
        positive = (1 << (bit_width - 1)) - 1
        code_min, code_max = -positive, positive
        maximum_absolute = max(abs(lower), abs(upper))
        ideal_scale = maximum_absolute / code_max if code_max > 0 else 0.0
    else:
        code_min, code_max = 0, (1 << bit_width) - 1
        ideal_scale = (upper - lower) / (code_max - code_min)
    scale, unit, _ = quantize_scale(ideal_scale, tensor.fractional_bits)
    zero_scale = scale <= 0.0
    if zero_scale and tensor.zero_scale_policy == "clamp_to_min_scale":
        scale = unit
    zero_point = 0
    if tensor.scheme == "asymmetric" and scale > 0.0:
        zero_point = max(
            code_min,
            min(code_max, _round_half_away(code_min - lower / scale)),
        )
    return {
        "ideal_scale": ideal_scale,
        "quantized_scale": scale,
        "zero_point": zero_point,
        "code_min": code_min,
        "code_max": code_max,
        "zero_scale": zero_scale,
    }


def reference_column(
    values: np.ndarray,
    missing_mask: np.ndarray,
    tensor: TensorConfig,
    bit_width: int,
) -> dict[str, Any]:
    non_missing = np.asarray(values[~missing_mask], dtype=np.float64)
    qlo = float(
        np.quantile(
            non_missing,
            tensor.quantile_lower_permille / 1000.0,
            method="linear",
        )
    )
    qhi = float(
        np.quantile(
            non_missing,
            tensor.quantile_upper_permille / 1000.0,
            method="linear",
        )
    )
    params = _reference_params(qlo, qhi, tensor, bit_width)
    scale = float(params["quantized_scale"])
    zero_point = int(params["zero_point"])
    code_min = int(params["code_min"])
    code_max = int(params["code_max"])
    codes: list[int | None] = []
    dequantized: list[float | None] = []
    absolute_errors: list[float | None] = []
    relative_errors: list[float | None] = []
    over_tolerance: list[bool] = []
    for row, value_raw in enumerate(values):
        if bool(missing_mask[row]):
            codes.append(None)
            dequantized.append(None)
            absolute_errors.append(None)
            relative_errors.append(None)
            over_tolerance.append(False)
            continue
        value = float(value_raw)
        if scale == 0.0:
            preclip = 0
        else:
            preclip = _round_half_away(value / scale + zero_point)
        if tensor.saturation_strategy == "wrap":
            levels = code_max - code_min + 1
            code = ((preclip - code_min) % levels) + code_min
        else:
            code = max(code_min, min(code_max, preclip))
        reconstructed = (float(code) - zero_point) * scale
        error = abs(reconstructed - value)
        relative = None if value == 0.0 else error / abs(value)
        violation = error > tensor.tolerance if value == 0.0 else relative > tensor.tolerance
        codes.append(int(code))
        dequantized.append(float(reconstructed))
        absolute_errors.append(float(error))
        relative_errors.append(relative)
        over_tolerance.append(bool(violation))
    return {
        "lower_quantile": qlo,
        "upper_quantile": qhi,
        "params": params,
        "codes": codes,
        "dequantized": dequantized,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "over_tolerance": over_tolerance,
    }


def verify_active_width(
    *,
    values: np.ndarray,
    missing_mask: np.ndarray,
    tensor: TensorConfig,
    calibration: dict[str, Any],
    params: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    reference = reference_column(
        values, missing_mask, tensor, int(params["bit_width"])
    )
    quantile_delta = max(
        abs(float(calibration["lower_quantile"]) - reference["lower_quantile"]),
        abs(float(calibration["upper_quantile"]) - reference["upper_quantile"]),
    )
    scale_delta = abs(
        float(params["quantized_scale"])
        - float(reference["params"]["quantized_scale"])
    )
    zero_point_equal = int(params["zero_point"]) == int(
        reference["params"]["zero_point"]
    )
    code_mismatch_count = 0
    dequant_delta_max = 0.0
    error_delta_max = 0.0
    tolerance_mismatch_count = 0
    for row, element in enumerate(analysis["elements"]):
        ref_code = reference["codes"][row]
        if ref_code is None:
            if not bool(element["missing"]):
                code_mismatch_count += 1
            continue
        if bool(element["missing"]) or int(element["quantized_code"]) != int(ref_code):
            code_mismatch_count += 1
            continue
        dequant_delta_max = max(
            dequant_delta_max,
            abs(float(element["dequantized"]) - float(reference["dequantized"][row])),
        )
        error_delta_max = max(
            error_delta_max,
            abs(float(element["abs_error"]) - float(reference["absolute_errors"][row])),
        )
        if bool(element["over_tolerance"]) != bool(reference["over_tolerance"][row]):
            tolerance_mismatch_count += 1
    passed = (
        quantile_delta <= 1e-12
        and scale_delta <= 1e-15
        and zero_point_equal
        and code_mismatch_count == 0
        and dequant_delta_max <= 1e-12
        and error_delta_max <= 1e-12
        and tolerance_mismatch_count == 0
    )
    return {
        "reference_dtype": "float64",
        "reference_quantile_method": "numpy.quantile(method='linear')",
        "quantile_max_abs_delta": quantile_delta,
        "scale_abs_delta": scale_delta,
        "zero_point_equal": bool(zero_point_equal),
        "code_mismatch_count": int(code_mismatch_count),
        "dequant_max_abs_delta": dequant_delta_max,
        "abs_error_max_abs_delta": error_delta_max,
        "tolerance_branch_mismatch_count": int(tolerance_mismatch_count),
        "passed": bool(passed),
    }
