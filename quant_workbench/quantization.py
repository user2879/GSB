"""Per-layer percentile calibration and deterministic fixed-point requantization."""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

from .models import LayerConfig, WeightLayer
from .util import clean_float, round_half_away, round_significant, stage_cache_key

_FIXED_MIN = -(2**62)
_FIXED_MAX = 2**62


class DeadlineExceeded(RuntimeError):
    """Raised by cooperative compute loops when the interactive budget expires."""


def _check_budget(
    deadline: float | None, cancel_event: threading.Event | None
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DeadlineExceeded("cancelled")
    if deadline is not None and time.perf_counter() >= deadline:
        raise DeadlineExceeded("interactive deadline reached")


def _group_slices(size: int, group_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + group_size, size)) for start in range(0, size, group_size)]


def calibrate_layer(
    weight: WeightLayer,
    config: LayerConfig,
    *,
    significant_digits: int = 8,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Compute symmetric and asymmetric scales/zero points for every group."""

    _check_budget(deadline, cancel_event)
    bits = int(config.bits)
    group_size = int(config.group_size)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amin = 0
    amax = 2**bits - 1
    flat = np.asarray(weight.weights, dtype=np.float64).reshape(-1)
    size = int(flat.size)

    slices = _group_slices(size, group_size)
    symmetric_scales = np.zeros(len(slices), dtype=np.float64)
    asymmetric_scales = np.zeros_like(symmetric_scales)
    asymmetric_zero_points = np.zeros_like(symmetric_scales)
    group_ids = np.empty(size, dtype=np.int64)
    group_records: list[dict[str, Any]] = []
    for group_index, (start, end) in enumerate(slices):
        _check_budget(deadline, cancel_event)
        values = flat[start:end]
        group_ids[start:end] = group_index
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        max_abs = float(np.max(np.abs(values)))

        symmetric_scale = max_abs / float(qmax) if max_abs > 0.0 else 1.0
        symmetric_scale = round_significant(symmetric_scale, significant_digits)
        if symmetric_scale == 0.0:
            symmetric_scale = 10.0 ** (-significant_digits)

        asymmetric_range = maximum - minimum
        if math.isfinite(asymmetric_range) and asymmetric_range > 0.0:
            asymmetric_scale = asymmetric_range / float(amax - amin)
        else:
            asymmetric_scale = sys.float_info.max / float(amax - amin)
        asymmetric_scale = round_significant(asymmetric_scale, significant_digits)
        if asymmetric_scale == 0.0:
            asymmetric_scale = 10.0 ** (-significant_digits)
        zero_float = -minimum / asymmetric_scale
        zero_point = int(round_half_away(np.asarray(zero_float)).item())
        zero_point = max(amin, min(amax, zero_point))

        symmetric_scales[group_index] = symmetric_scale
        asymmetric_scales[group_index] = asymmetric_scale
        asymmetric_zero_points[group_index] = zero_point
        group_records.append(
            {
                "index": group_index,
                "start": start,
                "end": end,
                "count": end - start,
                "minimum": clean_float(minimum),
                "maximum": clean_float(maximum),
                "max_abs": clean_float(max_abs),
                "symmetric_scale": clean_float(symmetric_scale),
                "symmetric_zero_point": 0,
                "asymmetric_scale": clean_float(asymmetric_scale),
                "asymmetric_zero_point": zero_point,
            }
        )
        if progress is not None:
            progress(group_index + 1)

    cache_key = stage_cache_key(
        "calibration",
        weight.content_fingerprint,
        bits,
        group_size,
        significant_digits,
    )
    return {
        "schema": "quant-workbench-calibration-v1",
        "layer": weight.layer,
        "version": weight.version,
        "shape": list(weight.shape),
        "order": weight.order,
        "content_fingerprint": weight.content_fingerprint,
        "calibration_cache_key": cache_key,
        "cache_key": cache_key,
        "config_fields": ["bits", "group_size", "significant_digits"],
        "calibration_sample_ids": [],
        "parameters": {
            "bits": bits,
            "group_size": group_size,
            "significant_digits": significant_digits,
            "symmetric_code_min": qmin,
            "symmetric_code_max": qmax,
            "asymmetric_code_min": amin,
            "asymmetric_code_max": amax,
        },
        "groups": group_records,
        "symmetric_scales": [clean_float(value) for value in symmetric_scales.tolist()],
        "symmetric_zero_points": [0] * len(slices),
        "asymmetric_scales": [clean_float(value) for value in asymmetric_scales.tolist()],
        "asymmetric_zero_points": [int(value) for value in asymmetric_zero_points.tolist()],
        "group_ids": [int(value) for value in group_ids.tolist()],
        "point_count": size,
    }


def requantize_layer(
    weight: WeightLayer,
    calibration: dict[str, Any],
    config: LayerConfig,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Convert calibrated real groups into saturated fixed-point integers."""

    _check_budget(deadline, cancel_event)
    bits = int(config.bits)
    scheme = config.scheme
    fractional_bits = int(config.fractional_bits)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amin = 0
    amax = 2**bits - 1
    if scheme == "symmetric":
        scale_values = calibration["symmetric_scales"]
        zero_values = [0] * len(scale_values)
        code_min, code_max = qmin, qmax
    else:
        scale_values = calibration["asymmetric_scales"]
        zero_values = calibration["asymmetric_zero_points"]
        code_min, code_max = amin, amax

    flat = np.asarray(weight.weights, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(calibration["group_ids"], dtype=np.int64)
    scales = np.asarray(scale_values, dtype=np.float64)
    zeros = np.asarray(zero_values, dtype=np.int64)
    element_scales = scales[group_ids]
    element_zeros = zeros[group_ids]

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        preclip_float = flat / element_scales + element_zeros
    # Preserve sign/direction for pathological finite values while keeping the
    # pre-clip integer representable in JSON and int64.
    preclip_float = np.nan_to_num(
        preclip_float,
        nan=0.0,
        posinf=float(2**62),
        neginf=float(-(2**62)),
    )
    preclip_float = np.clip(preclip_float, -(2.0**62), 2.0**62)
    preclip_integers = round_half_away(preclip_float).astype(np.int64)
    clipped = np.clip(preclip_integers, code_min, code_max).astype(np.int64)

    # Float64 can represent powers of two exactly up to 2**1023.
    fixed_factor = math.ldexp(1.0, min(fractional_bits, 1023))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reconstructed = (clipped.astype(np.float64) - element_zeros) * element_scales
    reconstructed = np.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=sys.float_info.max,
        neginf=-sys.float_info.max,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_unclipped_float = reconstructed * fixed_factor
    fixed_unclipped_float = np.nan_to_num(
        fixed_unclipped_float,
        nan=0.0,
        posinf=float(_FIXED_MAX),
        neginf=float(_FIXED_MIN),
    )
    fixed_unclipped_float = np.clip(
        fixed_unclipped_float, float(_FIXED_MIN), float(_FIXED_MAX)
    )
    fixed_unclipped = round_half_away(fixed_unclipped_float)
    fixed_clipped = np.clip(fixed_unclipped, _FIXED_MIN, _FIXED_MAX).astype(np.int64)
    fixed_dequantized = fixed_clipped.astype(np.float64) / fixed_factor

    truncations: list[dict[str, Any]] = []
    saturation_count = 0
    fixed_truncation_count = 0
    for index in range(int(flat.size)):
        if index % 4096 == 0:
            _check_budget(deadline, cancel_event)
        raw_int = int(preclip_integers[index])
        clipped_int = int(clipped[index])
        if raw_int < clipped_int:
            direction = "low"
        elif raw_int > clipped_int:
            direction = "high"
        else:
            direction = "none"
        if direction != "none":
            saturation_count += 1
        fixed_raw = int(round_half_away(np.asarray(fixed_unclipped_float[index])).item())
        fixed_value = int(fixed_clipped[index])
        fixed_direction = "none"
        if fixed_raw < fixed_value:
            fixed_direction = "low"
        elif fixed_raw > fixed_value:
            fixed_direction = "high"
        if fixed_direction != "none":
            fixed_truncation_count += 1
        truncations.append(
            {
                "flat_index": index,
                "group": int(group_ids[index]),
                "direction": direction,
                "preclip_int": raw_int,
                "clipped_int": clipped_int,
                "fixed_direction": fixed_direction,
                "preclip_fixed_int": fixed_raw,
                "fixed_int": fixed_value,
            }
        )

    selected_groups = [
        {
            **group,
            "scale": clean_float(scales[index]),
            "zero_point": int(zeros[index]),
            "scheme": scheme,
        }
        for index, group in enumerate(calibration["groups"])
    ]
    quantization_key = stage_cache_key(
        "quantization",
        calibration["calibration_cache_key"],
        fractional_bits,
        scheme,
    )
    return {
        "schema": "quant-workbench-quantization-v1",
        "layer": weight.layer,
        "version": weight.version,
        "shape": list(weight.shape),
        "order": weight.order,
        "content_fingerprint": weight.content_fingerprint,
        "parameter_fingerprint": config.parameter_fingerprint,
        "config_fields": ["bits", "group_size", "fractional_bits", "scheme"],
        "calibration_config_fields": calibration["config_fields"],
        "calibration_sample_ids": [],
        "calibration_cache_key": calibration["calibration_cache_key"],
        "cache_key": quantization_key,
        "quantization_cache_key": quantization_key,
        "parameters": {
            **config.to_dict(),
            "significant_digits": calibration["parameters"]["significant_digits"],
            "code_min": code_min,
            "code_max": code_max,
            "fixed_scale_factor": fixed_factor,
            "rounding": "half_away_from_zero",
            "saturation": True,
        },
        "groups": selected_groups,
        "symmetric_scales": calibration["symmetric_scales"],
        "symmetric_zero_points": calibration["symmetric_zero_points"],
        "asymmetric_scales": calibration["asymmetric_scales"],
        "asymmetric_zero_points": calibration["asymmetric_zero_points"],
        "scales": [clean_float(value) for value in scales.tolist()],
        "zero_points": [int(value) for value in zeros.tolist()],
        "group_ids": calibration["group_ids"],
        "integers": [int(value) for value in clipped.tolist()],
        "preclip_integers": [int(value) for value in preclip_integers.tolist()],
        "affine_dequantized": [clean_float(value) for value in reconstructed.tolist()],
        "fixed_integers": [int(value) for value in fixed_clipped.tolist()],
        "dequantized": [clean_float(value) for value in fixed_dequantized.tolist()],
        "truncations": truncations,
        "saturation_count": saturation_count,
        "fixed_truncation_count": fixed_truncation_count,
        "point_count": int(flat.size),
    }


def quantize_layer(
    weight: WeightLayer,
    config: LayerConfig,
    *,
    significant_digits: int = 8,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Convenience API for calibrating and requantizing one layer in one call."""

    calibration = calibrate_layer(
        weight,
        config,
        significant_digits=significant_digits,
        deadline=deadline,
        cancel_event=cancel_event,
        progress=progress,
    )
    return requantize_layer(
        weight,
        calibration,
        config,
        deadline=deadline,
        cancel_event=cancel_event,
    )


def dequantized_matrix(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["dequantized"], dtype=np.float64).reshape(result["shape"])


def integer_matrix(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["integers"], dtype=np.int64).reshape(result["shape"])


def quantization_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Small stable slice used by summaries and deployment manifests."""

    return {
        "layer": result["layer"],
        "version": result["version"],
        "cache_key": result["cache_key"],
        "bits": result["parameters"]["bits"],
        "group_size": result["parameters"]["group_size"],
        "fractional_bits": result["parameters"]["fractional_bits"],
        "scheme": result["parameters"]["scheme"],
        "saturation_count": result["saturation_count"],
        "point_count": result["point_count"],
    }
