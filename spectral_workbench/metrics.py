"""Concentration error metrics and uncertainty coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CubeConfig
from .unmix import EndmemberModel, UnmixResult, concentration_error_components


@dataclass(frozen=True)
class FrameMetrics:
    per_fluorophore: Tuple[Dict[str, Any], ...]
    absolute_errors: Tuple[np.ndarray, ...]
    uncertainty_bounds: Tuple[np.ndarray, ...]
    first_violation: Optional[Dict[str, Any]]
    evaluated_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_fluorophore": list(self.per_fluorophore),
            "first_tolerance_violation": self.first_violation,
            "evaluated_pixels": self.evaluated_pixels,
        }


def _truth_maps(
    cube_config: CubeConfig,
    truth: Dict[str, Any],
    order: int,
    compensated: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    height = cube_config.height
    width = cube_config.width
    yy, xx = np.mgrid[0:height, 0:width]
    maps = np.zeros((order, height, width), dtype=np.float64)
    masks = np.zeros((order, height, width), dtype=bool)
    for k, blob in enumerate(truth["blobs"][:order]):
        center = blob["reference_center"] if compensated else blob["center"]
        cx, cy = float(center[0]), float(center[1])
        peak = float(blob["true_concentration"])
        values = peak * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (cube_config.blob_sigma_px**2)))
        maps[k] = values
        masks[k] = values >= (0.25 * peak)
    return maps, masks


def _uncertainty_maps(
    model: EndmemberModel,
    result: UnmixResult,
    quantization_upper_bound: float,
) -> np.ndarray:
    order, height, width = result.concentrations.shape
    q = float(quantization_upper_bound)
    eps = float(np.finfo(np.float32).eps)
    active = result.active_subset.astype(np.int64)
    condition_lookup = np.ones(256, dtype=np.float64)
    for subset_id, value in model.subset_condition.items():
        if subset_id < condition_lookup.size:
            condition_lookup[subset_id] = value
    active_condition = condition_lookup[np.clip(active, 0, 255)]
    y_norm = np.maximum(result.signal_norm, 1.0e-12)
    calibration = model.row_norms[:, None, None] * q
    band_count = model.endmembers.shape[0]
    residual = (
        model.row_norms[:, None, None]
        * result.residual[None, :, :]
        * (np.sqrt(float(band_count)) * 2.0)
    )
    conditioning = (
        active_condition[None, :, :]
        * (q / y_norm)[None, :, :]
        * np.abs(result.concentrations.astype(np.float64))
    )
    roundoff = 4.0 * eps * (np.abs(result.concentrations.astype(np.float64)) + 1.0)
    bounds = calibration + residual + conditioning + roundoff
    bounds[:, result.invalid_mask.astype(bool)] = 0.0
    return bounds.astype(np.float32)


def compute_frame_metrics(
    cube_config: CubeConfig,
    truth: Dict[str, Any],
    model: EndmemberModel,
    result: UnmixResult,
    *,
    frame_index: int,
    compensated: bool,
    tolerance: float,
) -> FrameMetrics:
    truth_maps, truth_masks = _truth_maps(cube_config, truth, model.order, compensated)
    estimates = result.concentrations.astype(np.float64)
    error = estimates - truth_maps
    absolute_error = np.abs(error)
    uncertainty = _uncertainty_maps(model, result, float(result.record["quantization_upper_bound_dn"]))
    valid = result.invalid_mask == 0
    per_fluorophore: List[Dict[str, Any]] = []
    absolute_errors: List[np.ndarray] = []
    uncertainty_bounds: List[np.ndarray] = []
    for k in range(model.order):
        selected = truth_masks[k] & valid
        if np.any(selected):
            errors = absolute_error[k, selected]
            bounds = uncertainty[k, selected].astype(np.float64)
            coverage = np.abs(error[k, selected]) <= bounds
            record = {
                "fluorophore": k,
                "name": model.names[k],
                "pixels": int(np.count_nonzero(selected)),
                "concentration_mean": float(np.mean(estimates[k, selected])),
                "truth_mean": float(np.mean(truth_maps[k, selected])),
                "absolute_error_mean": float(np.mean(errors)),
                "absolute_error_p95": float(np.percentile(errors, 95)),
                "over_tolerance_pixels": int(np.count_nonzero(errors > tolerance)),
                "over_tolerance_fraction": float(np.mean(errors > tolerance)),
                "uncertainty_mean": float(np.mean(bounds)),
                "uncertainty_p95": float(np.percentile(bounds, 95)),
                "interval_coverage_fraction": float(np.mean(coverage)),
            }
        else:
            errors = np.empty(0, dtype=np.float64)
            bounds = np.empty(0, dtype=np.float64)
            record = {
                "fluorophore": k,
                "name": model.names[k],
                "pixels": 0,
                "concentration_mean": None,
                "truth_mean": None,
                "absolute_error_mean": None,
                "absolute_error_p95": None,
                "over_tolerance_pixels": 0,
                "over_tolerance_fraction": 0.0,
                "uncertainty_mean": None,
                "uncertainty_p95": None,
                "interval_coverage_fraction": None,
            }
        per_fluorophore.append(record)
        absolute_errors.append(errors)
        uncertainty_bounds.append(bounds)

    bad_pixels = np.any(absolute_error > tolerance, axis=0) & valid
    first_violation: Optional[Dict[str, Any]] = None
    if np.any(bad_pixels):
        flat_index = int(np.flatnonzero(bad_pixels.ravel(order="C"))[0])
        y, x = np.unravel_index(flat_index, bad_pixels.shape)
        components = concentration_error_components(
            model,
            result,
            frame_index=frame_index,
            y=int(y),
            x=int(x),
        )
        components["truth_values"] = [float(truth_maps[k, y, x]) for k in range(model.order)]
        components["absolute_errors"] = [float(absolute_error[k, y, x]) for k in range(model.order)]
        components["tolerance"] = float(tolerance)
        first_violation = components
    return FrameMetrics(
        per_fluorophore=tuple(per_fluorophore),
        absolute_errors=tuple(absolute_errors),
        uncertainty_bounds=tuple(uncertainty_bounds),
        first_violation=first_violation,
        evaluated_pixels=int(np.count_nonzero(valid)),
    )


def aggregate_frame_metrics(
    model: EndmemberModel,
    frames: Sequence[FrameMetrics],
    tolerance: float,
) -> Dict[str, Any]:
    result: List[Dict[str, Any]] = []
    for k in range(model.order):
        errors = np.concatenate([frame.absolute_errors[k] for frame in frames if frame.absolute_errors[k].size])
        bounds = np.concatenate([frame.uncertainty_bounds[k] for frame in frames if frame.uncertainty_bounds[k].size])
        if errors.size:
            result.append(
                {
                    "fluorophore": k,
                    "name": model.names[k],
                    "pixels": int(errors.size),
                    "absolute_error_mean": float(np.mean(errors)),
                    "absolute_error_p95": float(np.percentile(errors, 95)),
                    "over_tolerance_pixels": int(np.count_nonzero(errors > tolerance)),
                    "over_tolerance_fraction": float(np.mean(errors > tolerance)),
                    "uncertainty_mean": float(np.mean(bounds)),
                    "uncertainty_p95": float(np.percentile(bounds, 95)),
                    "interval_coverage_fraction": float(np.mean(errors <= bounds)),
                }
            )
        else:
            result.append(
                {
                    "fluorophore": k,
                    "name": model.names[k],
                    "pixels": 0,
                    "absolute_error_mean": None,
                    "absolute_error_p95": None,
                    "over_tolerance_pixels": 0,
                    "over_tolerance_fraction": 0.0,
                    "uncertainty_mean": None,
                    "uncertainty_p95": None,
                    "interval_coverage_fraction": None,
                }
            )
    first = next((frame.first_violation for frame in frames if frame.first_violation is not None), None)
    return {
        "per_fluorophore": result,
        "first_tolerance_violation": first,
        "overall_absolute_error_mean": float(
            np.mean(np.concatenate([error for frame in frames for error in frame.absolute_errors if error.size]))
        )
        if any(error.size for frame in frames for error in frame.absolute_errors)
        else 0.0,
        "tolerance": float(tolerance),
    }
