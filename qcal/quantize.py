"""Fixed-point rescaling, saturation, and dequantization error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from .codes import E_INTERNAL, E_QUANT_CLIP, E_ZERO_SCALE
from .io_utils import combined_fingerprint, sha256_json, stable_round_half_away
from .models import FailureRecord, LayerQuantResult, QuantizationResult, WeightPack


class QuantizationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fixed_point_scale(raw_scale: float, fraction_bits: int) -> float:
    denominator = float(1 << int(fraction_bits))
    rounded = np.rint(np.asarray([raw_scale * denominator], dtype=np.float64))[0]
    scale = float(rounded) / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0 / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        raise QuantizationError(E_ZERO_SCALE, "fixed-point scale representation is invalid")
    return scale


def _round(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    normalized = mode.lower()
    if normalized in ("nearest_even", "round_half_even", "bankers"):
        return np.rint(values)
    if normalized in ("nearest_away", "round_half_away", "half_up"):
        return stable_round_half_away(values)
    if normalized in ("floor", "down"):
        return np.floor(values)
    if normalized in ("ceil", "up"):
        return np.ceil(values)
    if normalized in ("truncate", "toward_zero"):
        return np.trunc(values)
    raise QuantizationError(E_INTERNAL, f"unsupported rounding mode: {mode}")


def quantize_layer(
    pack: WeightPack,
    quant: LayerQuantResult,
    fixed_cfg: dict[str, Any],
    *,
    full_precision: bool,
) -> QuantizationResult:
    fixed_fraction_bits = int(fixed_cfg["fraction_bits"])
    rounding_mode = str(fixed_cfg.get("rounding_mode", "nearest_even"))
    saturation = str(fixed_cfg.get("saturation", "saturate")).lower()
    flat = np.ascontiguousarray(pack.weights, dtype=np.float64).reshape(-1)
    total = int(flat.size)
    if full_precision:
        dequantized = flat.copy()
        directions = [0] * total
        group_payload = [group.to_dict() for group in quant.groups]
        return QuantizationResult(
            layer=pack.layer,
            shape=[int(value) for value in pack.shape],
            bit_width=quant.bit_width,
            fixed_point_fraction_bits=fixed_fraction_bits,
            rounding_mode=rounding_mode,
            saturation=saturation,
            full_precision=True,
            integers=[None] * total,
            dequantized=[float(value) for value in dequantized.tolist()],
            clip_directions=directions,
            clipped_count=0,
            clip_rate=0.0,
            max_abs_error=0.0,
            mse=0.0,
            group_parameters=group_payload,
            metadata={
                "full_precision_whitelist": True,
                "intermediate_integer_representation": "bypassed",
                "error_metadata": {
                    "bit_width": quant.bit_width,
                    "fixed_point_fraction_bits": fixed_fraction_bits,
                    "rounding_mode": rounding_mode,
                },
            },
        )

    integers = np.zeros(total, dtype=np.int64)
    dequantized = np.zeros(total, dtype=np.float64)
    directions = np.zeros(total, dtype=np.int8)
    effective_groups: list[dict[str, Any]] = []
    for group in quant.groups:
        start = group.start
        end = group.end
        if start < 0 or end > total or end <= start:
            raise QuantizationError(E_INTERNAL, f"invalid group range [{start}, {end}) for {pack.layer}")
        scale = _fixed_point_scale(float(group.raw_scale), fixed_fraction_bits)
        if not np.isfinite(scale) or scale <= 0.0:
            raise QuantizationError(E_ZERO_SCALE, f"group {group.index} has a non-positive scale")
        values = flat[start:end]
        q_float = values / scale + float(group.zero_point)
        q_int = _round(q_float, rounding_mode)
        above = q_int > group.qmax
        below = q_int < group.qmin
        if saturation == "error" and bool(np.any(above) or np.any(below)):
            raise QuantizationError(E_QUANT_CLIP, f"group {group.index} would saturate")
        if saturation not in ("saturate", "error"):
            raise QuantizationError(E_INTERNAL, f"unsupported saturation strategy: {saturation}")
        q_clipped = np.clip(q_int, group.qmin, group.qmax)
        directions[start:end] = above.astype(np.int8) - below.astype(np.int8)
        integers[start:end] = q_clipped.astype(np.int64)
        dequantized[start:end] = (q_clipped - float(group.zero_point)) * scale
        effective_groups.append(
            {
                "index": group.index,
                "start": start,
                "end": end,
                "effective_scale": scale,
                "raw_scale": group.raw_scale,
                "scale_fixed_fraction_bits": fixed_fraction_bits,
                "scale_minimum_clamped": bool(np.rint(np.asarray([group.raw_scale * (1 << fixed_fraction_bits)]))[0] <= 0.0),
                "zero_point": group.zero_point,
                "qmin": group.qmin,
                "qmax": group.qmax,
                "scheme": group.scheme,
            }
        )

    errors = dequantized - flat
    abs_errors = np.abs(errors)
    clipped_count = int(np.count_nonzero(directions))
    group_payload = [group.to_dict() for group in quant.groups]
    return QuantizationResult(
        layer=pack.layer,
        shape=[int(value) for value in pack.shape],
        bit_width=quant.bit_width,
        fixed_point_fraction_bits=fixed_fraction_bits,
        rounding_mode=rounding_mode,
        saturation=saturation,
        full_precision=False,
        integers=[int(value) for value in integers.tolist()],
        dequantized=[float(value) for value in dequantized.tolist()],
        clip_directions=[int(value) for value in directions.tolist()],
        clipped_count=clipped_count,
        clip_rate=float(clipped_count) / float(total) if total else 0.0,
        max_abs_error=float(np.max(abs_errors)) if total else 0.0,
        mse=float(np.mean(errors * errors)) if total else 0.0,
        group_parameters=group_payload,
        metadata={
            "effective_groups": effective_groups,
            "error_metadata": {
                "bit_width": quant.bit_width,
                "fixed_point_fraction_bits": fixed_fraction_bits,
                "rounding_mode": rounding_mode,
                "saturation": saturation,
                "scheme": quant.scheme,
            },
            "clip_direction_convention": "-1 below qmin, +1 above qmax, 0 in range",
        },
    )


def quantization_param_fingerprint(
    quant: LayerQuantResult,
    fixed_cfg: dict[str, Any],
    full_precision: bool,
) -> str:
    return sha256_json(
        {
            "quantile_param_fingerprint": quant.param_fingerprint,
            "bit_width": quant.bit_width,
            "fixed_point_fraction_bits": int(fixed_cfg["fraction_bits"]),
            "rounding_mode": str(fixed_cfg.get("rounding_mode", "nearest_even")),
            "saturation": str(fixed_cfg.get("saturation", "saturate")),
            "full_precision": bool(full_precision),
        }
    )


def quantization_cache_key(
    quant_key: str,
    param_fingerprint: str,
    exchange_fingerprint: str,
) -> str:
    return combined_fingerprint(
        "fixed_point_rescale",
        quant_key,
        param_fingerprint,
        exchange_fingerprint,
    )


def rescale_config_fingerprint(
    layer_cfg: dict[str, Any],
    fixed_cfg: dict[str, Any],
    full_precision: bool,
) -> str:
    return sha256_json(
        {
            "bit_width": int(layer_cfg["bit_width"]),
            "fixed_point_fraction_bits": int(fixed_cfg["fraction_bits"]),
            "rounding_mode": str(fixed_cfg.get("rounding_mode", "nearest_even")),
            "saturation": str(fixed_cfg.get("saturation", "saturate")),
            "full_precision": bool(full_precision),
        }
    )


def quantization_result_from_dict(payload: dict[str, Any]) -> QuantizationResult:
    return QuantizationResult(
        layer=str(payload["layer"]),
        shape=[int(value) for value in payload["shape"]],
        bit_width=int(payload["bit_width"]),
        fixed_point_fraction_bits=int(payload["fixed_point_fraction_bits"]),
        rounding_mode=str(payload["rounding_mode"]),
        saturation=str(payload["saturation"]),
        full_precision=bool(payload["full_precision"]),
        integers=list(payload["integers"]),
        dequantized=[float(value) for value in payload["dequantized"]],
        clip_directions=[int(value) for value in payload["clip_directions"]],
        clipped_count=int(payload["clipped_count"]),
        clip_rate=float(payload["clip_rate"]),
        max_abs_error=float(payload["max_abs_error"]),
        mse=float(payload["mse"]),
        group_parameters=list(payload.get("group_parameters", [])),
        metadata=dict(payload.get("metadata", {})),
    )


def quantization_failure(layer: str, error: Exception) -> FailureRecord:
    code = getattr(error, "code", E_INTERNAL)
    return FailureRecord(
        layer=layer,
        stage="fixed_point_rescale",
        code=str(code),
        message=str(error),
    )
