"""Fixed-point quantization, fractional storage, and saturation clipping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .calibration import CalibrationResult
from .config import EffectiveLayerConfig
from .errors import E_LAYER_QUANTIZATION, E_SATURATION_EXCEEDED, PipelineError
from .inputs import WeightLayer
from .util import round_values


@dataclass
class QuantizationResult:
    data: dict[str, Any]
    fixed_ints: np.ndarray
    dequantized: np.ndarray
    base_dequantized: np.ndarray
    unclipped_dequantized: np.ndarray
    clipped_mask: np.ndarray

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "QuantizationResult":
        return cls(
            data=dict(data),
            fixed_ints=np.asarray(data["fixed_ints"], dtype=np.int64),
            dequantized=np.asarray(data["dequantized"], dtype=np.float64),
            base_dequantized=np.asarray(data["base_dequantized"], dtype=np.float64),
            unclipped_dequantized=np.asarray(data["unclipped_dequantized"], dtype=np.float64),
            clipped_mask=np.asarray(data["clipped_mask"], dtype=bool),
        )


def quantize_layer(
    layer: WeightLayer,
    calibration: CalibrationResult,
    config: EffectiveLayerConfig,
) -> QuantizationResult:
    try:
        weights = np.asarray(layer.weights, dtype=np.float64).reshape(-1)
        scales = np.asarray(calibration.scales, dtype=np.float64)
        zero_points = np.asarray(calibration.zero_points, dtype=np.float64)
        if weights.size != scales.size or weights.size != zero_points.size:
            raise ValueError("calibration arrays do not cover every weight")

        multiplier = config.fixed_multiplier
        qmin, qmax = config.qmin, config.qmax
        continuous = weights / scales + zero_points
        base_integers = round_values(continuous, config.rounding)
        raw_fixed = round_values(continuous * multiplier, config.rounding)
        lower_fixed = float(qmin * multiplier)
        upper_fixed = float(qmax * multiplier)
        below = raw_fixed < lower_fixed
        above = raw_fixed > upper_fixed
        clipped_mask = below | above
        if config.saturation == "error" and np.any(clipped_mask):
            first = int(np.flatnonzero(clipped_mask)[0])
            raise PipelineError(
                E_SATURATION_EXCEEDED,
                f"fixed value {int(raw_fixed[first])} at element {first} exceeds "
                f"[{int(lower_fixed)}, {int(upper_fixed)}] and saturation policy is 'error'",
                stage="quantization",
                layer=layer.name,
            )
        clipped_fixed = np.clip(raw_fixed, lower_fixed, upper_fixed)
        fixed_ints = clipped_fixed.astype(np.int64)
        dequantized = (fixed_ints.astype(np.float64) / multiplier - zero_points) * scales
        unclipped_dequantized = (raw_fixed / multiplier - zero_points) * scales
        base_dequantized = (base_integers - zero_points) * scales

        clipped_records: list[dict[str, Any]] = []
        for element_index in np.flatnonzero(clipped_mask).astype(int).tolist():
            direction = "negative" if below[element_index] else "positive"
            group = calibration.data["groups"][int(calibration.group_index[element_index])]
            clipped_records.append(
                {
                    "element_index": int(element_index),
                    "group_id": group["group_id"],
                    "direction": direction,
                    "fixed_value_before_clamp": int(raw_fixed[element_index]),
                    "fixed_value_after_clamp": int(fixed_ints[element_index]),
                    "quantized_integer_before_clamp": int(raw_fixed[element_index]),
                    "lower_fixed_bound": int(lower_fixed),
                    "upper_fixed_bound": int(upper_fixed),
                }
            )

        fixed_ints_list = fixed_ints.tolist()
        summary = {
            "minimum": int(np.min(fixed_ints)) if fixed_ints.size else 0,
            "maximum": int(np.max(fixed_ints)) if fixed_ints.size else 0,
            "nonzero_count": int(np.count_nonzero(fixed_ints)),
            "zero_count": int(fixed_ints.size - np.count_nonzero(fixed_ints)),
            "positive_count": int(np.count_nonzero(fixed_ints > 0)),
            "negative_count": int(np.count_nonzero(fixed_ints < 0)),
            "clipped_count": int(clipped_records.__len__()),
            "lower_bound": int(lower_fixed),
            "upper_bound": int(upper_fixed),
            "checksum": int(np.sum(fixed_ints, dtype=np.int64)) if fixed_ints.size else 0,
        }
        data = {
            "stage": "quantization",
            "layer": layer.name,
            "shape": list(layer.shape),
            "element_count": layer.numel,
            "content_fingerprint": layer.content_fingerprint,
            "parameters": config.quantization_parameters(),
            "quantization_profile": config.profile,
            "target_bits": config.target_bits,
            "fractional_bits": config.fractional_bits,
            "fixed_multiplier": multiplier,
            "qmin": qmin,
            "qmax": qmax,
            "qmin_fixed": int(lower_fixed),
            "qmax_fixed": int(upper_fixed),
            "rounding": config.rounding,
            "saturation": config.saturation,
            "fixed_ints": fixed_ints_list,
            "dequantized": dequantized.tolist(),
            "base_dequantized": base_dequantized.tolist(),
            "unclipped_dequantized": unclipped_dequantized.tolist(),
            "clipped_mask": clipped_mask.tolist(),
            "fixed_int_summary": summary,
            "clipped_elements": clipped_records,
            "clipped_count": len(clipped_records),
        }
        return QuantizationResult(
            data=data,
            fixed_ints=fixed_ints,
            dequantized=dequantized,
            base_dequantized=base_dequantized,
            unclipped_dequantized=unclipped_dequantized,
            clipped_mask=clipped_mask,
        )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            E_LAYER_QUANTIZATION,
            f"quantization failed for layer {layer.name!r}: {exc}",
            stage="quantization",
            layer=layer.name,
            exception_type=type(exc).__name__,
        ) from exc

