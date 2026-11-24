"""Deterministic group-wise percentile calibration and scale derivation."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .model import Layer, LayerQuantConfig
from .utils import clean_float, round_half_away_from_zero, stable_hash


QUANTILE_METADATA = {
    "interpolation": "linear_n_minus_1",
    "position_rule": "position=(permille/1000)*(n-1)",
    "tie_break_rule": "stable_ascending_value_then_original_index; equal values are interchangeable",
    "sort_order": "ascending_float64",
}


def percentile_value(sorted_values: Sequence[float], permille: int) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty group")
    if not (0 <= permille <= 1000):
        raise ValueError("permille must be in [0, 1000]")
    if len(sorted_values) == 1:
        return clean_float(sorted_values[0])
    position = (permille / 1000.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean_float(sorted_values[lower])
    fraction = position - lower
    value = float(sorted_values[lower]) + fraction * (
        float(sorted_values[upper]) - float(sorted_values[lower])
    )
    return clean_float(value)


def _scale_quantum(fractional_bits: int) -> float:
    return clean_float(2.0 ** (-fractional_bits))


def _effective_scale(
    base_scale: float, cfg: LayerQuantConfig
) -> tuple[float, float]:
    quantum = _scale_quantum(cfg.fractional_bits)
    if not math.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = quantum
    target = np.float64(base_scale) * np.float64(cfg.quantization_levels)
    units = max(1, round_half_away_from_zero(float(target / np.float64(quantum))))
    scale = clean_float(
        float(np.float64(max(quantum, units * quantum))), cfg.scale_effective_digits
    )
    return scale, quantum


def derive_profile(
    *,
    lower_value: float,
    upper_value: float,
    cfg: LayerQuantConfig,
) -> dict[str, Any]:
    """Derive exact and effective affine parameters for one value range."""
    lower = clean_float(lower_value)
    upper = clean_float(upper_value)
    minimum = min(lower, upper)
    maximum = max(lower, upper)
    qmin = cfg.qmin
    qmax = cfg.qmax
    if cfg.scheme == "symmetric":
        absolute = max(abs(minimum), abs(maximum))
        if absolute <= 0.0:
            scale_base = 1.0
        else:
            scale_base = clean_float(
                float(np.float64(absolute) / np.float64(qmax)), cfg.scale_effective_digits
            )
        zero_base = 0
    else:
        span = maximum - minimum
        if span <= 0.0:
            scale_base = 1.0
        else:
            scale_base = clean_float(
                float(np.float64(span) / np.float64(qmax - qmin)),
                cfg.scale_effective_digits,
            )
        if scale_base <= 0.0 or not math.isfinite(scale_base):
            scale_base = 1.0
        zero_real = ((qmin + qmax) / 2.0) - ((minimum + maximum) / 2.0) / scale_base
        zero_base = min(qmax, max(qmin, round_half_away_from_zero(zero_real)))
    scale, quantum = _effective_scale(scale_base, cfg)
    if cfg.scheme == "symmetric":
        zero = 0
    else:
        zero_real = ((qmin + qmax) / 2.0) - ((minimum + maximum) / 2.0) / scale
        zero = min(qmax, max(qmin, round_half_away_from_zero(zero_real)))
    return {
        "scale_base": clean_float(scale_base, cfg.scale_effective_digits),
        "zero_base": zero_base,
        "scale": clean_float(scale, cfg.scale_effective_digits),
        "zero": zero,
        "scale_quantum": quantum,
    }


def calibrate_group(
    layer: Layer,
    cfg: LayerQuantConfig,
    group_index: int,
    *,
    calibration_sample_ids: Sequence[str],
) -> dict[str, Any]:
    start = group_index * cfg.group_granularity
    end = min(start + cfg.group_granularity, layer.numel)
    values = [clean_float(value) for value in layer.weights[start:end]]
    sorted_values = sorted(values)
    lower_quantile = percentile_value(sorted_values, cfg.lower_permille)
    upper_quantile = percentile_value(sorted_values, cfg.upper_permille)
    minimum = clean_float(sorted_values[0])
    maximum = clean_float(sorted_values[-1])
    absolute_quantile_max = max(abs(lower_quantile), abs(upper_quantile))
    profile = derive_profile(lower_value=lower_quantile, upper_value=upper_quantile, cfg=cfg)
    full_profile = derive_profile(lower_value=minimum, upper_value=maximum, cfg=cfg)

    payload: dict[str, Any] = {
        "kind": "group_calibration",
        "layer": layer.name,
        "content_version": layer.content_version,
        "content_fingerprint": layer.content_fingerprint,
        "group_index": group_index,
        "start": start,
        "end": end,
        "num_points": end - start,
        "group_granularity": cfg.group_granularity,
        "numel": layer.numel,
        "scheme": cfg.scheme,
        "bit_width": cfg.bit_width,
        "percentile_lower_permille": cfg.lower_permille,
        "percentile_upper_permille": cfg.upper_permille,
        "percentile_metadata": dict(QUANTILE_METADATA),
        "sorted_values": sorted_values,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "minimum": minimum,
        "maximum": maximum,
        "absolute_quantile_max": clean_float(absolute_quantile_max),
        "scale_base": profile["scale_base"],
        "zero_base": profile["zero_base"],
        "scale": profile["scale"],
        "zero": profile["zero"],
        "scale_quantum": profile["scale_quantum"],
        "full_range_scale": full_profile["scale"],
        "full_range_zero": full_profile["zero"],
        "qmin": cfg.qmin,
        "qmax": cfg.qmax,
        "calibration_sample_ids": list(calibration_sample_ids),
        "role_fingerprint": stable_hash(
            {
                "layer": layer.content_fingerprint,
                "group_index": group_index,
                "config": cfg.fingerprint_payload(),
                "quantile_method": QUANTILE_METADATA,
                "scale": profile["scale"],
                "zero": profile["zero"],
            }
        ),
    }
    return payload
