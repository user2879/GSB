"""Dark-current and flat-field correction with auditable error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import CalibrationConfig
from .errors import ErrorCode, WorkbenchError
from .io_utils import canonical_json, key_for, sha256_array, sha256_text


@dataclass(frozen=True)
class CalibrationResult:
    corrected: np.ndarray
    invalid_mask: np.ndarray
    saturated_pixels: int
    zero_flat_pixels: int
    invalid_pixels: int
    quantization_upper_bound: float
    quantization_mean_bound: float
    gain_min: float
    gain_max: float
    corrected_sha256: str
    invalid_sha256: str
    record: Dict[str, Any]

    def record_bytes(self) -> bytes:
        return (canonical_json(self.record) + "\n").encode("utf-8")


def _input_hash(raw: np.ndarray, dark: np.ndarray, flat: np.ndarray) -> str:
    return key_for(
        {
            "raw": sha256_array(np.ascontiguousarray(raw, dtype="<u2")),
            "dark": sha256_array(np.ascontiguousarray(dark, dtype="<u2")),
            "flat": sha256_array(np.ascontiguousarray(flat, dtype="<u2")),
        }
    )


def calibration_key(raw: np.ndarray, dark: np.ndarray, flat: np.ndarray, config: CalibrationConfig) -> str:
    return key_for(
        {
            "stage": "calibration-v1",
            "input": _input_hash(raw, dark, flat),
            "config": config.hash_payload(),
        }
    )


def calibrate_frame(
    raw: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    config: CalibrationConfig,
    *,
    frame_index: int = 0,
) -> CalibrationResult:
    """Correct one frame using ``(raw-dark) / ((flat-dark)/mean(flat-dark))``.

    Invalid pixels are replaced by a declared value, but are always retained in
    ``invalid_mask`` and counted in the deterministic record.
    """
    arrays = (raw, dark, flat)
    if any(array.ndim != 2 for array in arrays):
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "calibration inputs must be 2-D", {"frame": frame_index})
    if raw.shape != dark.shape or raw.shape != flat.shape:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "calibration input shape mismatch", {"frame": frame_index})
    if any(array.dtype != np.uint16 for array in arrays):
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "calibration inputs must be uint16", {"frame": frame_index})

    raw_f = raw.astype(np.float64)
    dark_f = dark.astype(np.float64)
    flat_f = flat.astype(np.float64)
    floor_delta = flat_f - dark_f

    dark_max = float(np.max(dark_f))
    if dark_max > config.max_dark_dn:
        raise WorkbenchError(
            ErrorCode.E_DARK_GAIN,
            "dark frame exceeds declared consistency bound",
            {"frame": frame_index, "dark_max": dark_max, "limit": config.max_dark_dn},
        )

    zero_flat = floor_delta <= config.flat_epsilon_dn
    saturated = raw >= 65535
    valid = (~zero_flat) & (~saturated)
    zero_count = int(np.count_nonzero(zero_flat))
    saturated_count = int(np.count_nonzero(saturated))
    pixel_count = int(raw.size)
    if zero_count / pixel_count > config.max_zero_flat_fraction:
        raise WorkbenchError(
            ErrorCode.E_DARK_GAIN,
            "zero/non-positive flat-field fraction exceeds policy",
            {"frame": frame_index, "zero_flat_pixels": zero_count, "fraction": zero_count / pixel_count},
        )
    if saturated_count / pixel_count > config.max_saturation_fraction:
        raise WorkbenchError(
            ErrorCode.E_CUBE_FORMAT,
            "saturated fraction exceeds declared input policy",
            {"frame": frame_index, "saturated_pixels": saturated_count, "fraction": saturated_count / pixel_count},
        )
    if not np.any(valid):
        raise WorkbenchError(ErrorCode.E_DARK_GAIN, "no valid flat-field pixels", {"frame": frame_index})

    mean_delta = float(np.mean(floor_delta[valid]))
    if not np.isfinite(mean_delta) or mean_delta <= config.flat_epsilon_dn:
        raise WorkbenchError(
            ErrorCode.E_DARK_GAIN,
            "mean flat-minus-dark signal is non-positive",
            {"frame": frame_index, "mean_flat_minus_dark": mean_delta},
        )

    safe_delta = np.where(valid, floor_delta, 1.0)
    gain = safe_delta / mean_delta
    corrected64 = (raw_f - dark_f) / gain
    corrected64[~valid] = config.invalid_value
    corrected = np.ascontiguousarray(corrected64, dtype=np.float32)
    invalid_mask = np.ascontiguousarray(~valid, dtype=np.uint8)

    # First-order worst-case propagation of independent +/-0.5 DN quantizers.
    abs_gain = np.abs(gain)
    abs_delta = np.abs(safe_delta)
    abs_corr = np.abs(corrected64)
    quant = 0.5 * (
        2.0 / abs_gain
        + (2.0 * abs_corr) / abs_delta
        + abs_corr / abs(mean_delta)
    )
    quant += 4.0 * float(np.finfo(np.float32).eps) * np.maximum(abs_corr, 1.0)
    quant[~valid] = 0.0
    q_max = float(np.max(quant[valid]))
    q_mean = float(np.mean(quant[valid]))

    record: Dict[str, Any] = {
        "formula_version": "dark-flat-v1",
        "frame": frame_index,
        "input_sha256": _input_hash(raw, dark, flat),
        "config_sha256": sha256_text(canonical_json(config.hash_payload())),
        "corrected_sha256": sha256_array(corrected),
        "invalid_mask_sha256": sha256_array(invalid_mask),
        "dtype": "float32-le",
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "saturated_pixels": saturated_count,
        "zero_flat_pixels": zero_count,
        "invalid_pixels": int(np.count_nonzero(~valid)),
        "quantization_upper_bound_dn": q_max,
        "quantization_mean_bound_dn": q_mean,
        "dark_max_dn": dark_max,
        "mean_flat_minus_dark_dn": mean_delta,
        "gain_min": float(np.min(gain[valid])),
        "gain_max": float(np.max(gain[valid])),
        "invalid_replacement": float(config.invalid_value),
    }
    return CalibrationResult(
        corrected=corrected,
        invalid_mask=invalid_mask,
        saturated_pixels=saturated_count,
        zero_flat_pixels=zero_count,
        invalid_pixels=int(np.count_nonzero(~valid)),
        quantization_upper_bound=q_max,
        quantization_mean_bound=q_mean,
        gain_min=float(np.min(gain[valid])),
        gain_max=float(np.max(gain[valid])),
        corrected_sha256=sha256_array(corrected),
        invalid_sha256=sha256_array(invalid_mask),
        record=record,
    )
