"""Deterministic per-layer, per-group percentile calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import EffectiveLayerConfig
from .errors import E_LAYER_CALIBRATION, PipelineError
from .inputs import WeightLayer


@dataclass
class CalibrationResult:
    data: dict[str, Any]
    scales: np.ndarray
    zero_points: np.ndarray
    group_index: np.ndarray

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CalibrationResult":
        numel = int(np.prod(data["shape"], dtype=np.int64))
        scales = np.empty(numel, dtype=np.float64)
        zero_points = np.empty(numel, dtype=np.int64)
        group_index = np.empty(numel, dtype=np.int64)
        for index, group in enumerate(data["groups"]):
            start = int(group["element_start"])
            end = int(group["element_end"])
            scales[start:end] = float(group["scale"])
            zero_points[start:end] = int(group["zero_point"])
            group_index[start:end] = index
        return cls(data=dict(data), scales=scales, zero_points=zero_points, group_index=group_index)


def quantile_sorted(values: np.ndarray, probability: float) -> float:
    """Linear quantile with a fixed (n-1)*p interpolation rule."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("empty quantile input")
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = np.sort(values, kind="mergesort")
    if ordered.size == 1:
        return float(ordered[0])
    position = (ordered.size - 1) * float(probability)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction)


def build_groups(
    layer_name: str,
    numel: int,
    group_size: int,
    min_group_size: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(0, numel, group_size)):
        end = min(start + group_size, numel)
        group_id = f"{layer_name}#g{ordinal:04d}"
        groups.append(
            {
                "group_id": group_id,
                "element_start": start,
                "element_end": end,
                "element_count": end - start,
                "source_group_ids": [group_id],
                "merged": False,
                "merge_basis": None,
            }
        )

    index = 0
    while index < len(groups):
        group = groups[index]
        if int(group["element_count"]) >= min_group_size or len(groups) == 1:
            index += 1
            continue
        if index > 0:
            previous = groups[index - 1]
            previous["element_end"] = group["element_end"]
            previous["element_count"] = int(previous["element_end"]) - int(previous["element_start"])
            previous["source_group_ids"].extend(group["source_group_ids"])
            previous["merged"] = True
            previous["merge_basis"] = (
                f"group {group['group_id']} had {group['element_count']} elements below "
                f"min_group_size={min_group_size}; merged into previous group"
            )
            groups.pop(index)
        else:
            following = groups[index + 1]
            group["element_end"] = following["element_end"]
            group["element_count"] = int(group["element_end"]) - int(group["element_start"])
            group["source_group_ids"].extend(following["source_group_ids"])
            group["merged"] = True
            group["merge_basis"] = (
                f"group was below min_group_size={min_group_size} and had no previous group; "
                f"merged with following group {following['group_id']}"
            )
            groups.pop(index + 1)
    return groups


def calibrate_layer(layer: WeightLayer, config: EffectiveLayerConfig) -> CalibrationResult:
    try:
        weights = np.asarray(layer.weights, dtype=np.float64).reshape(-1)
        if weights.size != layer.numel:
            raise ValueError("weight size mismatch")
        groups = build_groups(
            layer.name,
            layer.numel,
            config.group_size,
            config.min_group_size,
        )
        scales = np.empty(layer.numel, dtype=np.float64)
        zero_points = np.empty(layer.numel, dtype=np.int64)
        group_index = np.empty(layer.numel, dtype=np.int64)
        group_records: list[dict[str, Any]] = []
        qmin, qmax = config.qmin, config.qmax

        for index, group in enumerate(groups):
            start = int(group["element_start"])
            end = int(group["element_end"])
            values = weights[start:end]
            low_value = quantile_sorted(values, config.percentile_low)
            high_value = quantile_sorted(values, config.percentile_high)
            degenerate = False
            if config.scheme == "symmetric":
                absolute_maximum = max(abs(low_value), abs(high_value))
                if absolute_maximum == 0.0:
                    scale = 1.0
                    degenerate = True
                else:
                    scale = absolute_maximum / float(qmax)
                zero_point = 0
                scale_source = {
                    "kind": "symmetric_absolute_maximum_quantile",
                    "absolute_maximum": absolute_maximum,
                    "qmax": qmax,
                }
            else:
                span = high_value - low_value
                if span == 0.0:
                    scale = max(abs(low_value), 1.0) / float(qmax)
                    zero_point = 0
                    degenerate = True
                else:
                    scale = span / float(qmax - qmin)
                    zero_exact = qmin - low_value / scale
                    zero_point = int(np.clip(int(round(zero_exact)), qmin, qmax))
                scale_source = {
                    "kind": "asymmetric_min_max_quantile",
                    "span": span,
                    "qmin": qmin,
                    "qmax": qmax,
                }
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"derived non-positive scale for group {group['group_id']}")
            scales[start:end] = scale
            zero_points[start:end] = zero_point
            group_index[start:end] = index
            record = dict(group)
            record.update(
                {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "percentile_low_value": low_value,
                    "percentile_high_value": high_value,
                    "scale": float(scale),
                    "zero_point": zero_point,
                    "degenerate": degenerate,
                    "scale_source": scale_source,
                }
            )
            group_records.append(record)

        data = {
            "stage": "calibration",
            "layer": layer.name,
            "shape": list(layer.shape),
            "element_count": layer.numel,
            "content_fingerprint": layer.content_fingerprint,
            "parameters": config.calibration_parameters(),
            "quantization_profile": config.profile,
            "target_bits": config.target_bits,
            "qmin": qmin,
            "qmax": qmax,
            "percentile_metadata": {
                "low": config.percentile_low,
                "high": config.percentile_high,
                "interpolation": config.interpolation,
                "interpolation_rule": "sorted_values[(n-1)*p] with linear interpolation between adjacent ranks",
                "tie_break": config.tie_break,
                "tie_break_rule": "ascending index after stable mergesort; lower index wins for equal values",
                "quantile_method": "linear_n_minus_1",
            },
            "min_group_size": config.min_group_size,
            "requested_group_size": config.group_size,
            "groups": group_records,
            "scales": scales.tolist(),
            "zero_points": zero_points.tolist(),
        }
        return CalibrationResult(
            data=data,
            scales=scales,
            zero_points=zero_points,
            group_index=group_index,
        )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            E_LAYER_CALIBRATION,
            f"calibration failed for layer {layer.name!r}: {exc}",
            stage="calibration",
            layer=layer.name,
            exception_type=type(exc).__name__,
        ) from exc

