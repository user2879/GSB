"""Per-group fixed-point requantization, saturation and error attribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import GroupCalibration
from .hashing import fingerprint
from .models import LayerQuantConfig, level_range

ERR_DISCRET = "quantile_discretization"
ERR_SCALE = "scale_quantization"
ERR_SAT = "saturation"
ERR_ROUND = "dequant_rounding"
ERR_ZERO_BRANCH = "zero_truth_absolute_branch"


@dataclass
class GroupQuantResult:
    group_index: int
    raw_integers: list[int]
    integers: list[int]
    saturated: list[dict[str, Any]]
    dequant: list[float]
    abs_error: list[float]
    rel_error: list[float]
    error_kind: list[str]
    zero_truth_branch: list[bool]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "raw_integers": self.raw_integers,
            "integers": self.integers,
            "saturated": self.saturated,
            "dequant": self.dequant,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "error_kind": self.error_kind,
            "zero_truth_branch": self.zero_truth_branch,
            "fingerprint": self.fingerprint,
        }


def _round(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "trunc":
        return np.trunc(values)
    if mode == "floor":
        return np.floor(values)
    if mode == "half_even":
        return np.round(values)
    shifted = np.floor(values + 0.5) if mode == "half_up" else np.ceil(values - 0.5)
    return shifted


def _saturate(raw: np.ndarray, low: int, high: int,
              policy: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if policy == "clip":
        clipped = np.minimum(high, np.maximum(low, raw))
        for index, value in enumerate(raw):
            if value < low:
                records.append({"index": index, "direction": "low",
                                "pre_clip_integer": int(value)})
            elif value > high:
                records.append({"index": index, "direction": "high",
                                "pre_clip_integer": int(value)})
        return clipped.astype(np.int64), records
    if policy != "wrap":
        raise ValueError(f"unknown saturation policy: {policy}")
    span = high - low + 1
    wrapped = ((raw.astype(np.int64) - low) % span) + low
    for index, value in enumerate(raw):
        if value < low:
            records.append({"index": index, "direction": "low_wrap",
                            "pre_clip_integer": int(value)})
        elif value > high:
            records.append({"index": index, "direction": "high_wrap",
                            "pre_clip_integer": int(value)})
    return wrapped.astype(np.int64), records


def quantize_group(values: np.ndarray, calibration: GroupCalibration,
                   cfg: LayerQuantConfig) -> GroupQuantResult:
    low, high = level_range(cfg.level)
    grid = 2.0 ** cfg.frac_bits
    # Standard fixed-point flow: continuous affine value, grid rounding, clip.
    continuous = values / calibration.scale + calibration.zero
    raw = _round(continuous * grid, cfg.rounding).astype(np.int64)
    integers, saturation_records = _saturate(raw, low, high, cfg.saturation)

    # Fixed operation order: integer grid -> scale -> zero point.
    dequant = (integers / grid - calibration.zero) * calibration.scale

    abs_error = np.abs(dequant - values)
    rel_error = np.empty_like(abs_error)
    zero_branch = values == 0.0
    nonzero = ~zero_branch
    rel_error[nonzero] = abs_error[nonzero] / np.abs(values[nonzero])
    # True value is zero: relative error is undefined, absolute error is used.
    rel_error[zero_branch] = abs_error[zero_branch]

    error_kind = _attribute(values, continuous, raw, integers, dequant,
                            saturation_records, grid)
    material = {
        "cal_fp": calibration.fingerprint,
        "integers": integers.tolist(),
        "dequant": [float(x) for x in dequant],
        "level": cfg.level,
        "frac_bits": cfg.frac_bits,
        "rounding": cfg.rounding,
        "saturation": cfg.saturation,
    }
    return GroupQuantResult(
        group_index=calibration.group_index,
        raw_integers=[int(x) for x in raw],
        integers=[int(x) for x in integers],
        saturated=saturation_records,
        dequant=[float(x) for x in dequant],
        abs_error=[float(x) for x in abs_error],
        rel_error=[float(x) for x in rel_error],
        error_kind=error_kind,
        zero_truth_branch=[bool(x) for x in zero_branch],
        fingerprint=fingerprint(material, prefix="qnt-"),
    )


def _attribute(values: np.ndarray, continuous: np.ndarray, raw: np.ndarray,
               integers: np.ndarray, dequant: np.ndarray,
               saturation_records: list[dict[str, Any]],
               grid: float) -> list[str]:
    saturated_index = {record["index"] for record in saturation_records}
    kinds: list[str] = []
    rounded_continuous = np.round(continuous)
    for index in range(values.size):
        if index in saturated_index:
            kinds.append(ERR_SAT)
            continue
        residual = continuous[index] - rounded_continuous[index]
        if abs(residual) > 1.0 / grid:
            kinds.append(ERR_DISCRET)
            continue
        reconstructed = np.round(dequant[index], 12)
        if reconstructed != dequant[index]:
            kinds.append(ERR_ROUND)
            continue
        kinds.append(ERR_SCALE)
    return kinds


def assemble_dequant_matrix(layer_shape: tuple[int, ...],
                            group_results: list[GroupQuantResult]) -> np.ndarray:
    flat = np.concatenate([np.asarray(r.dequant, dtype=np.float64)
                           for r in group_results])
    return flat.reshape(layer_shape)


def requant_only_matrix(layer_shape: tuple[int, ...],
                        group_results: list[GroupQuantResult],
                        calibrations: list[GroupCalibration],
                        cfg: LayerQuantConfig) -> np.ndarray:
    """Dequantized values after integer rounding but before saturation."""
    grid = 2.0 ** cfg.frac_bits
    pieces = []
    for result, calibration in zip(group_results, calibrations):
        raw = np.asarray(result.raw_integers, dtype=np.float64)
        pieces.append((raw / grid - calibration.zero) * calibration.scale)
    return np.concatenate(pieces).reshape(layer_shape)
