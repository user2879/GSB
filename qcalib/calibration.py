"""Per-layer, per-group quantile calibration and scale/zero derivation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_ZERO_SCALE, FailureList
from .models import LayerQuantConfig, WeightLayer, level_range

INTERPOLATION = "linear_cdf_index"
TIE_RULE = "stable_mergesort_first_occurrence"
ZERO_SCALE_FALLBACK = 1.0


@dataclass
class GroupCalibration:
    group_index: int
    q_lo_value: float
    q_hi_value: float
    scale: float
    zero: int
    symmetric: bool
    zero_scale_fallback: bool
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "q_lo_value": self.q_lo_value,
            "q_hi_value": self.q_hi_value,
            "scale": self.scale,
            "zero": self.zero,
            "symmetric": self.symmetric,
            "zero_scale_fallback": self.zero_scale_fallback,
            "fingerprint": self.fingerprint,
        }


def quantile(sorted_values: np.ndarray, permille: int) -> float:
    """Deterministic quantile with a fixed interpolation/tie rule.

    ``permille`` is in [0, 1000]. The position uses the CDF index
    ``h = p * (n - 1)``; the two bracketing order statistics are linearly
    interpolated. Equal values are never reordered because a stable merge
    sort is applied upstream, so ties resolve to the first occurrence.
    """
    n = sorted_values.size
    if n == 1:
        return float(sorted_values[0])
    position = (permille / 1000.0) * (n - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    low_value = float(sorted_values[lower])
    high_value = float(sorted_values[upper])
    return low_value + (high_value - low_value) * fraction


def split_groups(weights: np.ndarray, group_size: int) -> list[np.ndarray]:
    flat = np.ascontiguousarray(weights, dtype=np.float64).ravel()
    if flat.size % group_size != 0:
        raise ValueError("group_size must divide the layer element count")
    return [flat[start:start + group_size]
            for start in range(0, flat.size, group_size)]


def calibrate_group(group_index: int, values: np.ndarray,
                    cfg: LayerQuantConfig,
                    failures: FailureList) -> GroupCalibration:
    from .hashing import array_fingerprint, fingerprint

    ordered = np.sort(values, kind="mergesort")
    q_lo_value = quantile(ordered, cfg.q_lo)
    q_hi_value = quantile(ordered, cfg.q_hi)
    low_int, high_int = level_range(cfg.level)
    symmetric = cfg.scheme == "sym"
    fallback = False

    if symmetric:
        magnitude = max(abs(q_lo_value), abs(q_hi_value))
        grid_half = (2.0 ** (cfg.frac_bits + cfg.bits - 1)) - 1.0
        if magnitude == 0.0:
            failures.add(E_ZERO_SCALE, "calibration",
                         f"{cfg.name}:g{group_index}",
                         "symmetric scale collapsed to zero",
                         f"fallback={ZERO_SCALE_FALLBACK}")
            scale = ZERO_SCALE_FALLBACK
            fallback = True
        else:
            scale = magnitude / grid_half
        zero = 0
    else:
        span = q_hi_value - q_lo_value
        grid_span = (2.0 ** (cfg.frac_bits + cfg.bits)) - 1.0
        if span <= 0.0:
            failures.add(E_ZERO_SCALE, "calibration",
                         f"{cfg.name}:g{group_index}",
                         "asymmetric span collapsed to zero",
                         f"fallback={ZERO_SCALE_FALLBACK}")
            scale = ZERO_SCALE_FALLBACK
            fallback = True
            zero = int(round(-(q_lo_value / scale)))
        else:
            scale = span / grid_span
            zero = int(round(low_int - q_lo_value / scale))
        zero = min(high_int, max(low_int, zero))

    group_fp = array_fingerprint(values, prefix="g-")
    material = {
        "group_fp": group_fp,
        "group_index": group_index,
        "q_lo_value": q_lo_value,
        "q_hi_value": q_hi_value,
        "scale": scale,
        "zero": zero,
        "scheme": cfg.scheme,
        "level": cfg.level,
        "q_lo": cfg.q_lo,
        "q_hi": cfg.q_hi,
    }
    return GroupCalibration(
        group_index=group_index,
        q_lo_value=q_lo_value,
        q_hi_value=q_hi_value,
        scale=float(scale),
        zero=zero,
        symmetric=symmetric,
        zero_scale_fallback=fallback,
        fingerprint=fingerprint(material, prefix="cal-"),
    )


def calibrate_layer(layer: WeightLayer, cfg: LayerQuantConfig,
                    failures: FailureList) -> list[GroupCalibration]:
    if layer.size % cfg.group_size != 0:
        raise ValueError(
            f"group_size {cfg.group_size} does not divide layer {layer.name}")
    groups = split_groups(layer.weights, cfg.group_size)
    return [calibrate_group(index, group, cfg, failures)
            for index, group in enumerate(groups)]


def calibration_metadata() -> dict[str, str]:
    return {
        "interpolation": INTERPOLATION,
        "tie_rule": TIE_RULE,
        "zero_scale_fallback": str(ZERO_SCALE_FALLBACK),
        "sort_kind": "mergesort",
        "integer_grid": "round((x/scale + zero) * 2^frac_bits)",
        "scale_convention": "quantiles mapped onto fractional fixed-point grid",
    }
