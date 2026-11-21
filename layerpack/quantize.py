"""Deterministic per-layer symmetric/asymmetric fixed-point quantization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import EffectiveLayerSettings
from .fingerprint import (
    normalize_float,
    ordered_abs_max,
    ordered_error_stats,
    ordered_min_max,
    sha256_array,
    sha256_json,
    to_builtin,
    truncate_significant,
)
from .model import Layer, QuantResult


def compute_quant_key(layer: Layer, settings: EffectiveLayerSettings) -> str:
    return sha256_json(
        {
            "kind": "quant-parameters-v1",
            "layer_content_fingerprint": layer.content_fingerprint,
            "settings": settings.fingerprint_payload(),
        }
    )


def compute_dequant_key(layer: Layer, settings: EffectiveLayerSettings, quant_key: str) -> str:
    return sha256_json(
        {
            "kind": "dequantized-weights-v1",
            "quant_parameters_key": quant_key,
            "layer_content_fingerprint": layer.content_fingerprint,
            "settings": settings.fingerprint_payload(),
        }
    )


def _normalize_axis(axis: int, ndim: int) -> int:
    resolved = axis if axis >= 0 else ndim + axis
    if resolved < 0 or resolved >= ndim:
        raise ValueError(f"group axis {axis} is out of range for {ndim} dimensions")
    return resolved


def _group_statistics(
    groups: np.ndarray, per_channel: bool
) -> tuple[list[float], list[float]]:
    """Return group min/max arrays in fixed channel-then-group order."""

    channel_count, group_count, _ = groups.shape
    minima = [0.0] * (channel_count * group_count)
    maxima = [0.0] * (channel_count * group_count)
    for channel in range(channel_count):
        for group_index in range(group_count):
            values = groups[channel, group_index]
            minimum = float(values[0])
            maximum = float(values[0])
            for item in values[1:]:
                number = float(item)
                if number < minimum:
                    minimum = number
                if number > maximum:
                    maximum = number
            minima[channel * group_count + group_index] = minimum
            maxima[channel * group_count + group_index] = maximum

    if not per_channel:
        combined_minima: list[float] = []
        combined_maxima: list[float] = []
        for group_index in range(group_count):
            minimum = minima[group_index]
            maximum = maxima[group_index]
            for channel in range(1, channel_count):
                index = channel * group_count + group_index
                if minima[index] < minimum:
                    minimum = minima[index]
                if maxima[index] > maximum:
                    maximum = maxima[index]
            combined_minima.append(minimum)
            combined_maxima.append(maximum)
        return combined_minima, combined_maxima
    return minima, maxima


def _fallback_parameters(
    weights: np.ndarray, bits: int, scheme: str, significant_digits: int
) -> tuple[float, int]:
    if scheme == "symmetric":
        qmax = float((1 << (bits - 1)) - 1)
        maximum = ordered_abs_max(weights)
        scale = truncate_significant(maximum / qmax, significant_digits) if maximum else 0.0
        if scale == 0.0:
            scale = normalize_float(1.0 / qmax)
        return scale, 0

    qmax = float((1 << bits) - 1)
    minimum, maximum = ordered_min_max(weights)
    scale = (
        truncate_significant((maximum - minimum) / qmax, significant_digits)
        if maximum != minimum
        else 0.0
    )
    if scale == 0.0:
        scale = normalize_float(1.0 / qmax)
        return scale, 0
    zero_point = int(math.floor((-minimum / scale) + 0.5))
    zero_point = min(max(zero_point, 0), int(qmax))
    return scale, zero_point


def _zero_point(minimum: float, scale: float, qmax: int) -> int:
    if scale == 0.0:
        return 0
    value = int(math.floor((-minimum / scale) + 0.5))
    return min(max(value, 0), qmax)


def _truncated_group_scales(
    minima: list[float],
    maxima: list[float],
    *,
    bits: int,
    scheme: str,
    significant_digits: int,
) -> tuple[list[float], list[int], list[int]]:
    scales: list[float] = []
    zero_points: list[int] = []
    degenerate: list[int] = []
    if scheme == "symmetric":
        qmax = float((1 << (bits - 1)) - 1)
        for index, (minimum, maximum) in enumerate(zip(minima, maxima)):
            absolute_max = max(abs(minimum), abs(maximum))
            raw_scale = absolute_max / qmax if absolute_max else 0.0
            scale = truncate_significant(raw_scale, significant_digits)
            if scale == 0.0:
                scale = 0.0
                degenerate.append(index)
            scales.append(scale)
            zero_points.append(0)
        return scales, zero_points, degenerate

    qmax_int = (1 << bits) - 1
    qmax = float(qmax_int)
    for index, (minimum, maximum) in enumerate(zip(minima, maxima)):
        raw_scale = (maximum - minimum) / qmax if maximum != minimum else 0.0
        scale = truncate_significant(raw_scale, significant_digits)
        if scale == 0.0:
            degenerate.append(index)
            scales.append(0.0)
            zero_points.append(0)
        else:
            scales.append(scale)
            zero_points.append(_zero_point(minimum, scale, qmax_int))
    return scales, zero_points, degenerate


def _expand_scales(
    values: list[float], channel_count: int, group_count: int, per_channel: bool
) -> np.ndarray:
    if per_channel:
        return np.asarray(values, dtype=np.float64).reshape(channel_count, group_count)
    return np.broadcast_to(
        np.asarray(values, dtype=np.float64)[None, :], (channel_count, group_count)
    ).copy()


def _expand_zero_points(
    values: list[int], channel_count: int, group_count: int, per_channel: bool
) -> np.ndarray:
    if per_channel:
        return np.asarray(values, dtype=np.int64).reshape(channel_count, group_count)
    return np.broadcast_to(
        np.asarray(values, dtype=np.int64)[None, :], (channel_count, group_count)
    ).copy()


def _parameters_payload(
    *,
    layer: Layer,
    settings: EffectiveLayerSettings,
    channel_count: int,
    group_count: int,
    group_size: int,
    axis: int,
    scale_values: list[float],
    zero_values: list[int],
    degenerate_indices: list[int],
    fallback_scale: float,
    fallback_zero: int,
    scheme_candidates: dict[str, Any],
) -> dict[str, Any]:
    if settings.per_channel:
        scales: Any = [
            scale_values[start : start + group_count]
            for start in range(0, len(scale_values), group_count)
        ]
        zero_points: Any = [
            zero_values[start : start + group_count]
            for start in range(0, len(zero_values), group_count)
        ]
        scale_shape = [channel_count, group_count]
    else:
        scales = scale_values
        zero_points = zero_values
        scale_shape = [group_count]

    sources = [
        {
            "group_index": index,
            "reason": "zero_scale_after_significant_digit_truncation",
            "source": "whole_layer",
        }
        for index in degenerate_indices
    ]
    payload: dict[str, Any] = {
        "layer": layer.name,
        "scheme": settings.scheme,
        "bits": settings.bits,
        "full_precision": settings.full_precision,
        "group_size": settings.group_size,
        "group_axis": settings.group_axis,
        "resolved_group_axis": axis,
        "group_count": group_count,
        "channel_count": channel_count,
        "per_channel_scaling": settings.per_channel,
        "scale_significant_digits": settings.scale_significant_digits,
        "scale_shape": scale_shape,
        "scales": [normalize_float(value) for value in scales] if not settings.per_channel else [
            [normalize_float(value) for value in row] for row in scales
        ],
        "zero_points": zero_points,
        "degenerated_group_count": len(degenerate_indices),
        "degeneration_sources": sources,
        "scheme_candidates": scheme_candidates,
        "whole_layer_fallback_scale": normalize_float(fallback_scale),
        "whole_layer_fallback_zero_point": fallback_zero,
        "rounding_mode": "nearest_ties_to_even",
        "group_accumulation_order": "original_c_order",
        "config_fields": sorted(
            [
                "quantization.default_bits",
                "quantization.scheme",
                "quantization.group_size",
                "quantization.group_axis",
                "quantization.per_channel_scaling",
                "quantization.scale_significant_digits",
                "full_precision_layers",
                f"layers.{layer.name}.bits",
                f"layers.{layer.name}.scheme",
                f"layers.{layer.name}.group_size",
                f"layers.{layer.name}.group_axis",
                f"layers.{layer.name}.per_channel",
            ]
        ),
        "layer_content_fingerprint": layer.content_fingerprint,
    }
    payload["parameters_sha256"] = sha256_json(to_builtin(payload))
    return payload


def _scheme_candidates(
    *,
    layer: Layer,
    settings: EffectiveLayerSettings,
    minima: list[float],
    maxima: list[float],
    channel_count: int,
    group_count: int,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for scheme in ("symmetric", "asymmetric"):
        scales, zero_points, degenerate = _truncated_group_scales(
            minima,
            maxima,
            bits=settings.bits,
            scheme=scheme,
            significant_digits=settings.scale_significant_digits,
        )
        fallback_scale, fallback_zero = _fallback_parameters(
            layer.weights,
            settings.bits,
            scheme,
            settings.scale_significant_digits,
        )
        for index in degenerate:
            scales[index] = fallback_scale
            zero_points[index] = fallback_zero
        if settings.per_channel:
            formatted_scales: Any = [
                [normalize_float(value) for value in scales[start : start + group_count]]
                for start in range(0, len(scales), group_count)
            ]
            formatted_zeros: Any = [
                zero_points[start : start + group_count]
                for start in range(0, len(zero_points), group_count)
            ]
            shape = [channel_count, group_count]
        else:
            formatted_scales = [normalize_float(value) for value in scales]
            formatted_zeros = zero_points
            shape = [group_count]
        candidates[scheme] = {
            "scales": formatted_scales,
            "zero_points": formatted_zeros,
            "scale_shape": shape,
            "degenerated_group_count": len(degenerate),
            "whole_layer_fallback_scale": normalize_float(fallback_scale),
            "whole_layer_fallback_zero_point": fallback_zero,
        }
    return candidates


def quantize_layer(
    layer: Layer,
    settings: EffectiveLayerSettings,
    *,
    key: str | None = None,
) -> QuantResult:
    """Quantize and dequantize one layer using a fully deterministic path."""

    quant_key = key or compute_quant_key(layer, settings)
    dequant_key = compute_dequant_key(layer, settings, quant_key)
    if settings.full_precision:
        dequantized = np.array(layer.weights, dtype=np.float64, copy=True, order="C")
        zeros = np.zeros(layer.shape, dtype=np.int64)
        parameters: dict[str, Any] = {
            "layer": layer.name,
            "scheme": "full_precision",
            "bits": settings.bits,
            "full_precision": True,
            "group_size": settings.group_size,
            "group_axis": settings.group_axis,
            "resolved_group_axis": _normalize_axis(settings.group_axis, len(layer.shape)),
            "group_count": 0,
            "channel_count": 0,
            "per_channel_scaling": settings.per_channel,
            "scale_significant_digits": settings.scale_significant_digits,
            "scale_shape": [0],
            "scales": [],
            "zero_points": [],
            "degenerated_group_count": 0,
            "degeneration_sources": [],
            "scheme_candidates": {},
            "whole_layer_fallback_scale": 1.0,
            "whole_layer_fallback_zero_point": 0,
            "rounding_mode": "none",
            "group_accumulation_order": "original_c_order",
            "config_fields": [
                "full_precision_layers",
                f"layers.{layer.name}.full_precision",
            ],
            "layer_content_fingerprint": layer.content_fingerprint,
        }
        parameters["parameters_sha256"] = sha256_json(to_builtin(parameters))
        digest = sha256_array(dequantized)
        return QuantResult(
            layer_name=layer.name,
            parameters=parameters,
            dequantized=dequantized,
            codes=zeros,
            distortion={
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "sum_squared_error": 0.0,
                "mse": 0.0,
                "element_count": layer.element_count,
            },
            quantized_sha256=digest,
            dequantized_sha256=digest,
            key=quant_key,
            dequant_key=dequant_key,
        )

    axis = _normalize_axis(settings.group_axis, layer.weights.ndim)
    dimension = layer.shape[axis]
    if dimension % settings.group_size != 0:
        raise ValueError(
            f"group size {settings.group_size} does not divide dimension {dimension}"
        )
    group_count = dimension // settings.group_size
    moved = np.moveaxis(layer.weights, axis, -1)
    channel_shape = moved.shape[:-1]
    channel_count = int(np.prod(channel_shape, dtype=np.int64)) if channel_shape else 1
    groups = np.ascontiguousarray(
        moved.reshape(channel_count, group_count, settings.group_size)
    )
    minima, maxima = _group_statistics(groups, settings.per_channel)
    scale_values, zero_values, degenerate_indices = _truncated_group_scales(
        minima,
        maxima,
        bits=settings.bits,
        scheme=settings.scheme,
        significant_digits=settings.scale_significant_digits,
    )
    fallback_scale, fallback_zero = _fallback_parameters(
        layer.weights,
        settings.bits,
        settings.scheme,
        settings.scale_significant_digits,
    )
    for index in degenerate_indices:
        scale_values[index] = fallback_scale
        zero_values[index] = fallback_zero
    candidates = _scheme_candidates(
        layer=layer,
        settings=settings,
        minima=minima,
        maxima=maxima,
        channel_count=channel_count,
        group_count=group_count,
    )

    scales = _expand_scales(
        scale_values, channel_count, group_count, settings.per_channel
    )
    zero_points = _expand_zero_points(
        zero_values, channel_count, group_count, settings.per_channel
    )
    broadcast_scales = scales[:, :, None]
    broadcast_zeros = zero_points[:, :, None]

    if settings.scheme == "symmetric":
        qmin = -(1 << (settings.bits - 1))
        qmax = (1 << (settings.bits - 1)) - 1
        codes = np.rint(groups / broadcast_scales)
        codes = np.clip(codes, qmin, qmax).astype(np.int64)
        dequant_groups = codes.astype(np.float64) * broadcast_scales
    else:
        qmax = (1 << settings.bits) - 1
        codes = np.rint(groups / broadcast_scales) + broadcast_zeros
        codes = np.clip(codes, 0, qmax).astype(np.int64)
        dequant_groups = (codes.astype(np.float64) - broadcast_zeros) * broadcast_scales

    dequant_moved = dequant_groups.reshape(moved.shape)
    dequantized = np.ascontiguousarray(np.moveaxis(dequant_moved, -1, axis))
    code_moved = codes.reshape(moved.shape)
    code_array = np.ascontiguousarray(np.moveaxis(code_moved, -1, axis)).astype(np.int64)
    difference = dequantized - layer.weights
    distortion = ordered_error_stats(difference)

    parameters = _parameters_payload(
        layer=layer,
        settings=settings,
        channel_count=channel_count,
        group_count=group_count,
        group_size=settings.group_size,
        axis=axis,
        scale_values=scale_values,
        zero_values=zero_values,
        degenerate_indices=degenerate_indices,
        fallback_scale=fallback_scale,
        fallback_zero=fallback_zero,
        scheme_candidates=candidates,
    )
    return QuantResult(
        layer_name=layer.name,
        parameters=parameters,
        dequantized=dequantized,
        codes=code_array,
        distortion=distortion,
        quantized_sha256=sha256_array(code_array, dtype="<i8"),
        dequantized_sha256=sha256_array(dequantized),
        key=quant_key,
        dequant_key=dequant_key,
    )
