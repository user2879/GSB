"""Deterministic quantile calibration and scale/zero-point derivation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .codes import E_INTERNAL, E_QUANTILE_RANGE, E_ZERO_SCALE
from .io_utils import combined_fingerprint, sha256_json, stable_round_half_even
from .models import FailureRecord, GroupQuantParam, LayerQuantResult, WeightPack

SUPPORTED_SCHEMES = {"symmetric", "asymmetric", "affine"}


class QuantileError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sorted_quantile(
    sorted_values: np.ndarray,
    probability: float,
    interpolation: str = "linear",
    tie_break: str = "lower_index",
) -> float:
    """Quantile with a stable, explicitly documented interpolation rule."""

    values = np.asarray(sorted_values, dtype=np.float64)
    if values.size == 0:
        raise QuantileError(E_QUANTILE_RANGE, "cannot take a quantile of an empty group")
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise QuantileError(E_QUANTILE_RANGE, f"invalid probability {probability}")
    position = (values.size - 1) * float(probability)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = float(values[lower_index])
    upper_value = float(values[upper_index])
    if lower_index == upper_index or lower_value == upper_value:
        return lower_value
    fraction = position - lower_index
    if interpolation == "linear":
        return float(lower_value + (upper_value - lower_value) * fraction)
    if interpolation == "lower":
        return lower_value
    if interpolation == "higher":
        return upper_value
    if interpolation == "nearest":
        if fraction < 0.5:
            return lower_value
        if fraction > 0.5:
            return upper_value
        return lower_value if tie_break == "lower_index" else upper_value
    if interpolation == "midpoint":
        return float((lower_value + upper_value) * 0.5)
    raise QuantileError(E_QUANTILE_RANGE, f"unsupported interpolation rule: {interpolation}")


def _group_size(pack: WeightPack, layer_cfg: dict[str, Any], total: int) -> int:
    if bool(layer_cfg.get("per_channel", False)):
        if len(pack.shape) == 0:
            return total
        return max(1, int(pack.shape[-1]))
    requested = int(layer_cfg.get("group_size", 0) or 0)
    if requested <= 0 or requested >= total:
        return total
    return requested


def _calibrate_group(
    values: np.ndarray,
    group_index: int,
    start: int,
    end: int,
    *,
    bit_width: int,
    scheme: str,
    quantile_lower: float,
    quantile_upper: float,
    interpolation: str,
    tie_break: str,
) -> GroupQuantParam:
    qmin = -(1 << (bit_width - 1))
    qmax = (1 << (bit_width - 1)) - 1
    ordered = np.sort(values, kind="mergesort")
    q_low = sorted_quantile(ordered, quantile_lower, interpolation, tie_break)
    q_high = sorted_quantile(ordered, quantile_upper, interpolation, tie_break)
    observed_min = float(ordered[0])
    observed_max = float(ordered[-1])
    normalized_scheme = "asymmetric" if scheme == "affine" else scheme
    if normalized_scheme == "symmetric":
        max_abs = max(abs(q_low), abs(q_high))
        if not math.isfinite(max_abs) or max_abs <= 0.0:
            raise QuantileError(E_ZERO_SCALE, "symmetric quantile range is zero")
        raw_scale = float(max_abs) / float(qmax)
        zero_point = 0
    elif normalized_scheme == "asymmetric":
        if not (math.isfinite(q_low) and math.isfinite(q_high)):
            raise QuantileError(E_QUANTILE_RANGE, "quantile values are not finite")
        width = float(q_high - q_low)
        if width <= 0.0:
            raise QuantileError(E_QUANTILE_RANGE, "asymmetric quantile range must be positive")
        raw_scale = width / float(qmax - qmin)
        zero_raw = float(qmin) - float(q_low) / raw_scale
        zero_point = int(stable_round_half_even(np.asarray([zero_raw]))[0])
        zero_point = max(qmin, min(qmax, zero_point))
    else:
        raise QuantileError(E_QUANTILE_RANGE, f"unsupported quantization scheme: {scheme}")
    if not math.isfinite(raw_scale) or raw_scale <= 0.0:
        raise QuantileError(E_ZERO_SCALE, "derived scale is not positive and finite")
    return GroupQuantParam(
        index=group_index,
        start=start,
        end=end,
        q_low=float(q_low),
        q_high=float(q_high),
        observed_min=observed_min,
        observed_max=observed_max,
        raw_scale=float(raw_scale),
        scale=float(raw_scale),
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        scheme=normalized_scheme,
        interpolation=interpolation,
        tie_break=tie_break,
        fixed_fraction_bits=0,
    )


def calibrate_layer(
    pack: WeightPack,
    layer_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
    exchange_version: str,
) -> LayerQuantResult:
    bit_width = int(layer_cfg["bit_width"])
    if bit_width < 1 or bit_width > 32:
        raise QuantileError(E_QUANTILE_RANGE, f"bit width {bit_width} outside [1, 32]")
    scheme = str(layer_cfg.get("scheme", calibration_cfg.get("scheme", "asymmetric")))
    if scheme == "affine":
        scheme = "asymmetric"
    if scheme not in SUPPORTED_SCHEMES and scheme != "asymmetric":
        raise QuantileError(E_QUANTILE_RANGE, f"unsupported quantization scheme: {scheme}")
    lower = float(calibration_cfg["quantile_lower"])
    upper = float(calibration_cfg["quantile_upper"])
    interpolation = str(calibration_cfg.get("interpolation", "linear"))
    tie_break = str(calibration_cfg.get("tie_break", "lower_index"))
    flat = np.ascontiguousarray(pack.weights, dtype=np.float64).reshape(-1)
    total = int(flat.size)
    group_size = _group_size(pack, layer_cfg, total)
    groups: list[GroupQuantParam] = []
    for group_index, start in enumerate(range(0, total, group_size)):
        end = min(total, start + group_size)
        groups.append(
            _calibrate_group(
                flat[start:end],
                group_index,
                start,
                end,
                bit_width=bit_width,
                scheme=scheme,
                quantile_lower=lower,
                quantile_upper=upper,
                interpolation=interpolation,
                tie_break=tie_break,
            )
        )
    qmin = -(1 << (bit_width - 1))
    qmax = (1 << (bit_width - 1)) - 1
    parameter_payload = {
        "bit_width": bit_width,
        "group_size": group_size,
        "per_channel": bool(layer_cfg.get("per_channel", False)),
        "scheme": scheme,
        "quantile_lower": lower,
        "quantile_upper": upper,
        "interpolation": interpolation,
        "tie_break": tie_break,
    }
    param_fingerprint = sha256_json(parameter_payload)
    metadata = {
        "algorithm": "sorted_weight_quantiles",
        "sort": "numpy mergesort ascending",
        "interpolation": interpolation,
        "tie_break": tie_break,
        "quantile_lower": lower,
        "quantile_upper": upper,
        "group_size_rule": "per_channel_last_axis" if layer_cfg.get("per_channel") else "flattened_contiguous",
        "zero_point_rounding": "nearest_even_then_clip",
        "scale_fixed_point_rule": "applied later in fixed_point_rescale",
        "parameter_payload": parameter_payload,
    }
    return LayerQuantResult(
        layer=pack.layer,
        shape=[int(value) for value in pack.shape],
        bit_width=bit_width,
        qmin=qmin,
        qmax=qmax,
        group_size=group_size,
        per_channel=bool(layer_cfg.get("per_channel", False)),
        scheme=scheme,
        interpolation=interpolation,
        tie_break=tie_break,
        quantile_lower=lower,
        quantile_upper=upper,
        fixed_fraction_bits=0,
        groups=groups,
        content_fingerprint=pack.content_fingerprint,
        param_fingerprint=param_fingerprint,
        exchange_version=exchange_version,
        metadata=metadata,
    )


def layer_quant_key(
    pack: WeightPack,
    param_fingerprint: str,
    exchange_fingerprint: str,
) -> str:
    return combined_fingerprint(
        "quantile_calibration",
        pack.layer,
        pack.content_fingerprint,
        param_fingerprint,
        exchange_fingerprint,
    )


def quantile_parameter_fingerprint(
    layer_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
) -> str:
    scheme = str(layer_cfg.get("scheme", calibration_cfg.get("scheme", "asymmetric")))
    if scheme == "affine":
        scheme = "asymmetric"
    total_group_rule = (
        "per_channel_last_axis" if layer_cfg.get("per_channel") else "flattened_contiguous"
    )
    return sha256_json(
        {
            "bit_width": int(layer_cfg["bit_width"]),
            "group_size": int(layer_cfg.get("group_size", 0) or 0),
            "per_channel": bool(layer_cfg.get("per_channel", False)),
            "scheme": scheme,
            "quantile_lower": float(calibration_cfg["quantile_lower"]),
            "quantile_upper": float(calibration_cfg["quantile_upper"]),
            "interpolation": str(calibration_cfg.get("interpolation", "linear")),
            "tie_break": str(calibration_cfg.get("tie_break", "lower_index")),
            "group_rule": total_group_rule,
        }
    )


def layer_result_from_dict(payload: dict[str, Any]) -> LayerQuantResult:
    groups = [
        GroupQuantParam(
            index=int(item["index"]),
            start=int(item["start"]),
            end=int(item["end"]),
            q_low=float(item["q_low"]),
            q_high=float(item["q_high"]),
            observed_min=float(item["observed_min"]),
            observed_max=float(item["observed_max"]),
            raw_scale=float(item["raw_scale"]),
            scale=float(item["scale"]),
            zero_point=int(item["zero_point"]),
            qmin=int(item["qmin"]),
            qmax=int(item["qmax"]),
            scheme=str(item["scheme"]),
            interpolation=str(item["interpolation"]),
            tie_break=str(item["tie_break"]),
            fixed_fraction_bits=int(item.get("fixed_fraction_bits", 0)),
        )
        for item in payload["groups"]
    ]
    return LayerQuantResult(
        layer=str(payload["layer"]),
        shape=[int(value) for value in payload["shape"]],
        bit_width=int(payload["bit_width"]),
        qmin=int(payload["qmin"]),
        qmax=int(payload["qmax"]),
        group_size=int(payload["group_size"]),
        per_channel=bool(payload["per_channel"]),
        scheme=str(payload["scheme"]),
        interpolation=str(payload["interpolation"]),
        tie_break=str(payload["tie_break"]),
        quantile_lower=float(payload["quantile_lower"]),
        quantile_upper=float(payload["quantile_upper"]),
        fixed_fraction_bits=int(payload["fixed_fraction_bits"]),
        groups=groups,
        content_fingerprint=str(payload["content_fingerprint"]),
        param_fingerprint=str(payload["param_fingerprint"]),
        exchange_version=str(payload["exchange_version"]),
        metadata=dict(payload.get("metadata", {})),
    )


def quantile_failure(layer: str, error: Exception) -> FailureRecord:
    code = getattr(error, "code", E_INTERNAL)
    return FailureRecord(
        layer=layer,
        stage="quantile_calibration",
        code=str(code),
        message=str(error),
    )
