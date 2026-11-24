"""Axis mapping recovery, inverse mapping, and business-value quantization."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .util import sha256_json, stable_round
from .vision import StageError


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _fit_line(pixels: Sequence[float], transformed_values: Sequence[float], mode: str) -> Tuple[float, float, float]:
    if len(pixels) < 2:
        raise StageError("E_AXIS_MAPPING_FAILED", "at least two associated ticks are required")
    if mode == "endpoints":
        left_index = 0
        right_index = len(pixels) - 1
        x_left, x_right = transformed_values[left_index], transformed_values[right_index]
        y_left, y_right = pixels[left_index], pixels[right_index]
        if abs(x_right - x_left) < 1e-15:
            raise StageError("E_AXIS_MAPPING_FAILED", "endpoint fit has a degenerate value range")
        slope = (y_right - y_left) / (x_right - x_left)
        intercept = y_left - slope * x_left
    elif mode == "ols":
        count = len(pixels)
        mean_x = sum(transformed_values) / count
        mean_y = sum(pixels) / count
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(transformed_values, pixels))
        denominator = sum((x - mean_x) ** 2 for x in transformed_values)
        if abs(denominator) < 1e-15:
            raise StageError("E_AXIS_MAPPING_FAILED", "values have zero variance")
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
    else:
        raise StageError("E_AXIS_MAPPING_FAILED", f"unsupported inverse mode: {mode}")
    if abs(slope) < 1e-15:
        raise StageError("E_AXIS_MAPPING_FAILED", "fitted slope is zero")

    residuals = [pixel - (slope * value + intercept) for pixel, value in zip(pixels, transformed_values)]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return slope, intercept, rmse


def _transform_values(values: Sequence[float], transform: str) -> List[float]:
    if transform == "identity":
        return [float(value) for value in values]
    if transform == "log10":
        if any(value <= 0 for value in values):
            raise StageError("E_AXIS_MAPPING_FAILED", "log10 mapping requires positive tick values")
        return [math.log10(float(value)) for value in values]
    raise StageError("E_AXIS_MAPPING_FAILED", f"unsupported value transform: {transform}")


def _inverse_value(pixel: float, slope: float, intercept: float, transform: str) -> float:
    transformed = (pixel - intercept) / slope
    if transform == "identity":
        return transformed
    if transform == "log10":
        exponent = transformed * math.log(10.0)
        if exponent > 700:
            raise StageError("E_AXIS_MAPPING_FAILED", "inverse log value overflow")
        if exponent < -745:
            return 0.0
        return math.exp(exponent)
    raise StageError("E_AXIS_MAPPING_FAILED", f"unsupported value transform: {transform}")


def _fit_segment(
    pairs: Sequence[Dict[str, Any]],
    transform: str,
    inverse_mode: str,
) -> Dict[str, Any]:
    positions = [float(pair["position"]) for pair in pairs]
    values = [float(pair["value"]) for pair in pairs]
    transformed = _transform_values(values, transform)
    slope, intercept, residual = _fit_line(positions, transformed, inverse_mode)
    pixel_min, pixel_max = min(positions), max(positions)
    span = max(1.0, pixel_max - pixel_min)
    return {
        "transform": transform,
        "slope": stable_round(slope),
        "intercept": stable_round(intercept),
        "residual": stable_round(residual),
        "normalized_residual": stable_round(residual / span),
        "pixel_range": [int(min(positions)), int(max(positions))],
        "value_range": [stable_round(min(values)), stable_round(max(values))],
        "tick_ids": [pair["tick_id"] for pair in pairs],
        "tick_positions": [int(pair["position"]) for pair in pairs],
        "tick_values": [stable_round(pair["value"]) for pair in pairs],
    }


def _mapping_pairs(axis_data: Dict[str, Any], axis: str) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    ticks = axis_data["ticks"]
    associations = axis_data["associations"]
    by_tick = {association["tick_id"]: association for association in associations}
    pairs: List[Dict[str, Any]] = []
    for tick in ticks:
        association = by_tick.get(tick["id"])
        if association is None:
            continue
        pairs.append(
            {
                "tick_id": tick["id"],
                "position": int(tick["position"]),
                "value": float(association["value"]),
                "label_id": association["label_id"],
                "label_text": association["text"],
            }
        )
    pairs.sort(key=lambda pair: (pair["position"], pair["tick_id"]))
    associated_ids = {pair["tick_id"] for pair in pairs}
    excluded = [tick["id"] for tick in ticks if tick["id"] not in associated_ids]
    return pairs, excluded, associations


def _choose_break_index(pairs: Sequence[Dict[str, Any]], threshold: float) -> Optional[int]:
    if len(pairs) < 4:
        return None
    deltas = [abs(pairs[index + 1]["value"] - pairs[index]["value"]) for index in range(len(pairs) - 1)]
    pixel_deltas = [abs(pairs[index + 1]["position"] - pairs[index]["position"]) for index in range(len(pairs) - 1)]
    median_delta = _median(deltas)
    if median_delta <= 0:
        return None
    eligible = list(range(1, len(pairs) - 2))
    if not eligible:
        return None
    best_index = max(eligible, key=lambda index: (deltas[index], -index))
    ratio_excess = (deltas[best_index] / median_delta) - 1.0
    if ratio_excess < float(threshold):
        return None
    local_scales = [
        pixel_deltas[index] / deltas[index]
        for index in range(len(deltas))
        if deltas[index] > 0
    ]
    median_scale = _median(local_scales)
    if median_scale <= 0:
        return None
    scale_excess = abs(local_scales[best_index] / median_scale - 1.0)
    # A missing intermediate label can enlarge a value delta while preserving
    # a constant pixel/value slope. A genuine break changes both.
    if scale_excess < float(threshold) * 0.2:
        return None
    return best_index


def recover_axis_mapping(axis: str, axis_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ticks = axis_data["ticks"]
    pairs, excluded, associations = _mapping_pairs(axis_data, axis)
    if not ticks:
        raise StageError("E_AXIS_MAPPING_FAILED", f"{axis} axis has no ticks")
    excluded_ratio = len(excluded) / float(len(ticks))
    if excluded_ratio > float(config["label_occlusion_tolerance"]) + 1e-12:
        raise StageError(
            "E_LABEL_OCCLUSION",
            f"{axis} axis excludes {len(excluded)}/{len(ticks)} labels, above tolerance",
        )
    if len(pairs) < 2:
        raise StageError("E_AXIS_MAPPING_FAILED", f"{axis} axis has fewer than two associated ticks")

    ordered_values = [pair["value"] for pair in pairs]
    ordered_deltas = [ordered_values[index + 1] - ordered_values[index] for index in range(len(ordered_values) - 1)]
    if not (all(delta > 0 for delta in ordered_deltas) or all(delta < 0 for delta in ordered_deltas)):
        raise StageError("E_AXIS_LABEL_INCONSISTENT", f"{axis} labels are not monotonic along the axis")

    inverse_mode = str(config["inverse_mode"])
    linear_segment = _fit_segment(pairs, "identity", inverse_mode)
    selected_type = "linear"
    selected_transform = "identity"
    segments: List[Dict[str, Any]]

    log_segment: Optional[Dict[str, Any]] = None
    if len(pairs) >= 3 and all(value > 0 for value in ordered_values):
        log_segment = _fit_segment(pairs, "log10", inverse_mode)
        if (
            log_segment["normalized_residual"] <= float(config["log_axis_detection_threshold"]) + 1e-15
            and log_segment["normalized_residual"] + 1e-15 < linear_segment["normalized_residual"]
        ):
            selected_type = "log10"
            selected_transform = "log10"

    break_index = None
    if selected_type == "linear":
        break_index = _choose_break_index(pairs, float(config["broken_axis_detection_threshold"]))

    if break_index is not None:
        selected_type = "broken_linear"
        segments = [
            _fit_segment(pairs[: break_index + 1], "identity", inverse_mode),
            _fit_segment(pairs[break_index + 1 :], "identity", inverse_mode),
        ]
    elif selected_transform == "log10":
        segments = [log_segment]  # type: ignore[list-item]
    else:
        segments = [linear_segment]

    for index, segment in enumerate(segments):
        segment["segment_index"] = index

    mapping: Dict[str, Any] = {
        "axis": axis,
        "mapping_type": selected_type,
        "value_transform": selected_transform,
        "inverse_mode": inverse_mode,
        "segments": segments,
        "residual": stable_round(max(segment["residual"] for segment in segments)),
        "normalized_residual": stable_round(max(segment["normalized_residual"] for segment in segments)),
        "referenced_tick_ids": [pair["tick_id"] for pair in pairs],
        "referenced_tick_positions": [int(pair["position"]) for pair in pairs],
        "referenced_tick_values": [stable_round(pair["value"]) for pair in pairs],
        "excluded_tick_ids": excluded,
        "excluded_tick_count": len(excluded),
        "association_count": len(pairs),
        "tick_count": len(ticks),
        "associated_labels": [
            {
                "tick_id": pair["tick_id"],
                "label_id": pair["label_id"],
                "text": pair["label_text"],
                "value": stable_round(pair["value"]),
            }
            for pair in pairs
        ],
        "parameters": {
            "inverse_mode": inverse_mode,
            "log_axis_detection_threshold": float(config["log_axis_detection_threshold"]),
            "broken_axis_detection_threshold": float(config["broken_axis_detection_threshold"]),
            "label_occlusion_tolerance": float(config["label_occlusion_tolerance"]),
        },
    }
    mapping["mapping_id"] = sha256_json(mapping)
    return mapping


def _select_segment(mapping: Dict[str, Any], pixel: float) -> Tuple[Dict[str, Any], bool]:
    segments = mapping["segments"]
    if len(segments) == 1:
        return segments[0], False
    for segment in segments:
        low, high = segment["pixel_range"]
        if low <= pixel <= high:
            return segment, False
    candidates = []
    for segment in segments:
        low, high = segment["pixel_range"]
        distance = 0.0 if low <= pixel <= high else min(abs(pixel - low), abs(pixel - high))
        candidates.append((distance, segment["segment_index"], segment))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], True


def inverse_pixel(mapping: Dict[str, Any], pixel: float) -> Dict[str, Any]:
    segment, in_gap = _select_segment(mapping, pixel)
    value = _inverse_value(
        float(pixel),
        float(segment["slope"]),
        float(segment["intercept"]),
        str(segment["transform"]),
    )
    return {
        "raw_value": float(value),
        "segment_index": int(segment["segment_index"]),
        "selected_from_break_gap": bool(in_gap),
    }


def quantize_value(raw_value: float, step: float, decimal_places: int) -> Dict[str, Any]:
    raw_decimal = Decimal(str(raw_value))
    step_decimal = Decimal(str(step))
    if step_decimal > 0:
        quotient = (raw_decimal / step_decimal).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantized = quotient * step_decimal
    else:
        quantized = raw_decimal
    quantum = Decimal(1).scaleb(-int(decimal_places))
    fixed = quantized.quantize(quantum, rounding=ROUND_HALF_UP)
    if fixed > raw_decimal:
        direction = "up"
    elif fixed < raw_decimal:
        direction = "down"
    else:
        direction = "none"
    return {
        "value": float(fixed),
        "value_text": f"{fixed:.{int(decimal_places)}f}",
        "rounding_direction": direction,
        "raw_value": stable_round(raw_value, 12),
        "quantized_value": float(fixed),
        "quantization_step": float(step),
        "fixed_decimal_places": int(decimal_places),
    }


def restore_data_points(
    candidates: Sequence[Dict[str, Any]],
    chart_type: str,
    y_mapping: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    quota = int(config["page_data_point_quota"])
    errors: List[Dict[str, str]] = []
    selected = list(candidates)
    if quota < 0:
        quota = 0
    if len(selected) > quota:
        errors.append(
            {
                "code": "E_PAGE_QUOTA_EXCEEDED",
                "message": f"detected {len(selected)} data points; only the first {quota} are processed",
            }
        )
        selected = selected[:quota]
    if not selected:
        if not errors:
            errors.append({"code": "E_DATA_POINTS_NOT_FOUND", "message": "no chart data points were detected"})
        return {"chart_type": chart_type, "data_points": [], "errors": errors, "detected_count": len(candidates)}

    points: List[Dict[str, Any]] = []
    for sequence, candidate in enumerate(selected):
        pixel_x, pixel_y = int(candidate["pixel"][0]), int(candidate["pixel"][1])
        inverse = inverse_pixel(y_mapping, float(pixel_y))
        quantized = quantize_value(
            inverse["raw_value"],
            float(config["quantization_step"]),
            int(config["fixed_decimal_places"]),
        )
        point = {
            "sequence": sequence,
            "pixel": [pixel_x, pixel_y],
            "value": quantized["value"],
            "value_text": quantized["value_text"],
            "rounding_direction": quantized["rounding_direction"],
            "raw_value": quantized["raw_value"],
            "quantized_value": quantized["quantized_value"],
            "quantization_step": quantized["quantization_step"],
            "fixed_decimal_places": quantized["fixed_decimal_places"],
            "chart_type": chart_type,
            "component_id": int(candidate["component_id"]),
            "axis_mapping": {
                "axis": "y",
                "mapping_id": y_mapping["mapping_id"],
                "mapping_type": y_mapping["mapping_type"],
                "segment_index": inverse["segment_index"],
                "selected_from_break_gap": inverse["selected_from_break_gap"],
            },
        }
        points.append(point)
    return {
        "chart_type": chart_type,
        "data_points": points,
        "errors": errors,
        "detected_count": len(candidates),
    }
