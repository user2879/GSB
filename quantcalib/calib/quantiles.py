"""Per-group quantile calibration and scale / zero-point derivation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

_EPS = 1e-12


def fixed_quantile(sorted_values: np.ndarray, permille: int) -> tuple[float, int]:
    """Deterministic quantile with linear interpolation and tie rule.

    Returns the quantile value and the resolved rank. When the ideal position
    lands between two indices, linear interpolation is used; when the two
    bracketing values are equal (a tie), the lower index wins deterministically.
    """
    n = sorted_values.size
    if n == 1:
        return float(sorted_values[0]), 0
    pos = (permille / 1000.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo]), lo
    frac = pos - lo
    if float(sorted_values[hi]) == float(sorted_values[lo]):
        return float(sorted_values[lo]), lo
    value = float(sorted_values[lo]) + frac * (
        float(sorted_values[hi]) - float(sorted_values[lo])
    )
    return value, lo


def round_to_integer(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "floor":
        return np.floor(values)
    if mode == "half_even":
        return np.round(values)  # NumPy default is banker's rounding.
    # half_up: deterministic for ties, vectorized.
    floor = np.floor(values)
    diff = values - floor
    return floor + np.where(diff >= 0.5, 1.0, 0.0)


@dataclass
class GroupCalibration:
    layer: str
    group_index: int
    q_lo: float
    q_hi: float
    rank_lo: int
    rank_hi: int
    scale: float
    zero_point: int
    raw_step: float
    step_residual: float
    scheme: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_group(layer: str, group_index: int, values: np.ndarray,
                    bits: int, fractional_bits: int, lo_permille: int,
                    hi_permille: int, scheme: str) -> GroupCalibration:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    q_lo, rank_lo = fixed_quantile(ordered, lo_permille)
    q_hi, rank_hi = fixed_quantile(ordered, hi_permille)

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    grid = float(1 << fractional_bits)

    if scheme == "symmetric":
        amplitude = max(abs(q_lo), abs(q_hi))
        amplitude = max(amplitude, _EPS)
        raw_step = amplitude / qmax
        zero_point = 0
    else:
        span = max(q_hi - q_lo, _EPS)
        raw_step = span / (qmax - qmin)
        zero_float = qmin - q_lo / raw_step
        zero_point = int(np.clip(np.round(zero_float), qmin, qmax))

    # Snap the step onto the configured fixed-point grid so that switching
    # fractional_bits / quantization levels genuinely changes the integers.
    scale = max(np.round(raw_step * grid) / grid, 1.0 / grid)
    step_residual = float(scale - raw_step)

    return GroupCalibration(
        layer=layer,
        group_index=group_index,
        q_lo=float(q_lo),
        q_hi=float(q_hi),
        rank_lo=rank_lo,
        rank_hi=rank_hi,
        scale=float(scale),
        zero_point=zero_point,
        raw_step=float(raw_step),
        step_residual=step_residual,
        scheme=scheme,
    )
