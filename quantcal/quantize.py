"""Deterministic affine re-quantisation, saturation and dequantisation kernel."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import ResolvedTensorConfig
from .errors import QuantCalError
from .models import quant_bounds, symmetric_saturation_max
from .util import fingerprint, half_away_round


def _round_array(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "nearest_even":
        return np.rint(values)
    if mode == "nearest_away":
        return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))
    if mode == "floor":
        return np.floor(values)
    if mode == "ceil":
        return np.ceil(values)
    if mode == "truncate":
        return np.trunc(values)
    raise QuantCalError("E_FORMAT_INVALID", f"unsupported rounding mode {mode!r}", stage="quantize")


def effective_scale(scale: float, fractional_bits: int) -> tuple[float, int, str | None]:
    quantum = 2.0 ** (-fractional_bits)
    coefficient = half_away_round(scale / quantum)
    guard = None
    if coefficient <= 0:
        coefficient = 1
        guard = "effective_scale_clamped_to_one_quantum"
    return coefficient * quantum, coefficient, guard


def quantize_tensor(
    tensor_name: str,
    values: list[float],
    layout: Iterable[dict[str, Any]],
    resolved: ResolvedTensorConfig,
    scale_result: dict[str, Any],
    exchange_version: str,
) -> dict[str, Any]:
    if any(not math.isfinite(value) for value in values):
        raise QuantCalError(
            "E_NON_FINITE_SAMPLE", "non-finite value reached quantisation stage", tensor=tensor_name, stage="quantize"
        )
    bits = int(scale_result["quantization_bits"])
    scheme = str(scale_result["quantization_scheme"])
    scale = float(scale_result["scale"])
    zero_point = int(scale_result["zero_point"])
    qmin, qmax = quant_bounds(scheme, bits)
    if not (qmin <= zero_point <= qmax):
        raise QuantCalError(
            "E_ZERO_POINT_INVALID",
            "zero point outside exchange quantisation range",
            tensor=tensor_name,
            stage="quantize",
        )
    saturation_strategy = resolved.saturation_strategy
    saturation_max = int(scale_result.get("saturation_max", qmax))
    if saturation_max <= 0:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            "saturation max must be positive",
            tensor=tensor_name,
            stage="quantize",
        )
    if saturation_strategy == "saturate":
        clamp_min, clamp_max = qmin, min(qmax, saturation_max)
    elif saturation_strategy == "saturate_symmetric":
        symmetric_max = symmetric_saturation_max(bits)
        clamp_min, clamp_max = -symmetric_max, min(symmetric_max, saturation_max)
    else:
        raise QuantCalError(
            "E_FORMAT_INVALID",
            f"unsupported saturation strategy {saturation_strategy!r}",
            tensor=tensor_name,
            stage="quantize",
        )
    array = np.asarray(values, dtype=np.float64)
    used_scale, scale_coefficient, scale_guard = effective_scale(
        scale, resolved.fixed_fractional_bits
    )
    shifted = array / used_scale + float(zero_point)
    rounded = _round_array(shifted, resolved.rounding_mode)
    quantized = np.clip(rounded, clamp_min, clamp_max).astype(np.int64)
    clipped_mask = rounded != quantized
    dequantized = (quantized.astype(np.float64) - float(zero_point)) * used_scale
    absolute_error = np.abs(dequantized - array)
    layout_list = list(layout)
    clipped_records: list[dict[str, Any]] = []
    for flat_index in np.flatnonzero(clipped_mask):
        index = int(flat_index)
        direction = "upper" if rounded[index] > clamp_max else "lower"
        location = layout_list[index] if index < len(layout_list) else {}
        clipped_records.append(
            {
                "flat_index": index,
                "sample_id": location.get("sample_id"),
                "element_index": location.get("element_index"),
                "direction": direction,
                "value_before_clamp": int(rounded[index]),
                "value_after_clamp": int(quantized[index]),
            }
        )
    max_error_index = int(np.argmax(absolute_error)) if len(absolute_error) else 0
    scale_guard_reason = scale_result.get("scale_guard")
    if len(quantized) == 0:
        attribution = "no_elements"
    elif clipped_mask[max_error_index]:
        attribution = "saturation_clamp"
    elif scale_guard_reason:
        attribution = "calibration_zero_range_guard"
    elif exchange_version == "v1":
        attribution = "v1_milli_scale_quantization"
    elif scale_guard:
        attribution = "fixed_point_scale_quantization"
    else:
        attribution = "fixed_point_rounding"
    quantized_list = [int(value) for value in quantized.tolist()]
    dequantized_list = [float(value) for value in dequantized.tolist()]
    clipped_ratio = float(len(clipped_records)) / float(len(values)) if values else 0.0
    payload = {
        "tensor": tensor_name,
        "shape": None,
        "quantization_bits": bits,
        "quantization_scheme": scheme,
        "quantization_bounds": [qmin, qmax],
        "saturation_max": saturation_max,
        "saturation_strategy": saturation_strategy,
        "clamp_bounds": [clamp_min, clamp_max],
        "rounding_mode": resolved.rounding_mode,
        "fixed_fractional_bits": resolved.fixed_fractional_bits,
        "scale": scale,
        "effective_scale": used_scale,
        "effective_scale_coefficient": scale_coefficient,
        "effective_scale_guard": scale_guard,
        "zero_point": zero_point,
        "quantized_values": quantized_list,
        "dequantized_values": dequantized_list,
        "clipped_positions": clipped_records,
        "clipped_count": len(clipped_records),
        "clipped_element_ratio": clipped_ratio,
        "quantized_summary": {
            "count": len(quantized_list),
            "min": min(quantized_list) if quantized_list else None,
            "max": max(quantized_list) if quantized_list else None,
            "sum": int(sum(quantized_list)),
            "sha256": fingerprint(quantized_list),
        },
        "exchange_version": exchange_version,
        "error_attribution": {
            "stage": attribution,
            "max_error_flat_index": max_error_index,
            "max_error_was_clipped": bool(clipped_mask[max_error_index]) if len(quantized) else False,
            "scale_guard": scale_guard_reason,
        },
    }
    return payload


def precision_summary(quantize_result: dict[str, Any], values: list[float]) -> dict[str, Any]:
    if any(not math.isfinite(value) for value in values):
        raise QuantCalError(
            "E_NON_FINITE_SAMPLE",
            "non-finite value reached precision summary",
            tensor=quantize_result["tensor"],
            stage="precision",
        )
    original = np.asarray(values, dtype=np.float64)
    dequantized = np.asarray(quantize_result["dequantized_values"], dtype=np.float64)
    if original.shape != dequantized.shape:
        raise QuantCalError(
            "E_INTERNAL",
            "dequantized length differs from source length",
            tensor=quantize_result["tensor"],
            stage="precision",
        )
    absolute = np.abs(dequantized - original)
    squared = np.square(dequantized - original)
    max_index = int(np.argmax(absolute)) if len(absolute) else 0
    max_abs_error = float(absolute[max_index]) if len(absolute) else 0.0
    clipped_positions = quantize_result["clipped_positions"]
    max_was_clipped = any(item["flat_index"] == max_index for item in clipped_positions)
    return {
        "tensor": quantize_result["tensor"],
        "element_count": len(values),
        "clipped_count": int(quantize_result["clipped_count"]),
        "clipped_element_ratio": float(quantize_result["clipped_element_ratio"]),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(absolute.mean()) if len(absolute) else 0.0,
        "mse": float(squared.mean()) if len(squared) else 0.0,
        "max_error_flat_index": max_index,
        "max_error_is_clipped": bool(max_was_clipped),
        "error_attribution": (
            "saturation_clamp"
            if max_was_clipped
            else quantize_result["error_attribution"]["stage"]
        ),
        "quantized_values_sha256": quantize_result["quantized_summary"]["sha256"],
    }
