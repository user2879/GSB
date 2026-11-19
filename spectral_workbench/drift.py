"""Frame-to-frame drift estimation and subpixel compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .config import DriftConfig
from .errors import ErrorCode, WorkbenchError


@dataclass(frozen=True)
class DriftFrameResult:
    frame: int
    estimated_dx: float
    estimated_dy: float
    truth_dx: float
    truth_dy: float
    residual_dx: float
    residual_dy: float
    residual_norm: float
    max_estimated_norm: float
    centers_observed: Tuple[Tuple[float, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "estimated_shift_px": [self.estimated_dx, self.estimated_dy],
            "truth_shift_px": [self.truth_dx, self.truth_dy],
            "estimate_residual_px": [self.residual_dx, self.residual_dy],
            "estimate_residual_norm_px": self.residual_norm,
            "estimated_shift_norm_px": self.max_estimated_norm,
            "centers_observed_xy": [list(value) for value in self.centers_observed],
        }


@dataclass(frozen=True)
class DriftSequenceResult:
    frames: Tuple[DriftFrameResult, ...]
    max_residual_px: float
    rms_residual_px: float
    max_shift_px: float
    compensation_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "max_estimate_residual_px": self.max_residual_px,
            "rms_estimate_residual_px": self.rms_residual_px,
            "max_estimated_shift_px": self.max_shift_px,
            "compensation_enabled": self.compensation_applied,
        }


def _centroid_in_window(
    image: np.ndarray,
    reference_center: Tuple[float, float],
    radius: int,
    background_percentile: float,
    min_signal_dn: float,
) -> Optional[Tuple[float, float, float]]:
    height, width = image.shape
    cx, cy = reference_center
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(width, int(np.ceil(cx + radius + 1)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(height, int(np.ceil(cy + radius + 1)))
    crop = image[y0:y1, x0:x1].astype(np.float64)
    if crop.size == 0:
        return None
    floor = float(np.percentile(crop, background_percentile))
    peak = float(np.max(crop)) - floor
    if peak < min_signal_dn:
        return None
    weights = np.maximum(crop - floor, 0.0)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float(np.sum(weights * xx) / total), float(np.sum(weights * yy) / total), peak


def estimate_drift_sequence(
    calibrated: np.ndarray,
    invalid_mask: np.ndarray,
    reference_centers: Sequence[Sequence[float]],
    truth_shifts: Sequence[Sequence[float]],
    config: DriftConfig,
) -> DriftSequenceResult:
    stack = np.asarray(calibrated, dtype=np.float64)
    invalid = np.asarray(invalid_mask, dtype=np.uint8)
    if stack.ndim != 3 or invalid.shape != stack.shape:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "drift input must be a band stack with matching invalid mask")
    if len(reference_centers) == 0:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "at least one reference center is required")
    if len(truth_shifts) != stack.shape[0]:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "truth shift count does not match frames")

    references = tuple((float(value[0]), float(value[1])) for value in reference_centers)
    if len(references) > 1:
        spacing = min(
            float(np.hypot(references[i][0] - references[j][0], references[i][1] - references[j][1]))
            for i in range(len(references))
            for j in range(i + 1, len(references))
        )
        effective_radius = max(3, min(config.window_radius_px, int(0.4 * spacing)))
    else:
        effective_radius = config.window_radius_px
    frame_results = []
    residuals = []
    max_shift = 0.0
    for frame in range(stack.shape[0]):
        integrated = np.where(invalid[frame] != 0, 0.0, stack[frame])
        measured: list[Tuple[float, float]] = []
        qualities: list[float] = []
        measured_references: list[Tuple[float, float]] = []
        observed_centers: list[Tuple[float, float]] = []
        for center in references:
            centroid = _centroid_in_window(
                integrated,
                center,
                effective_radius,
                config.background_percentile,
                config.min_center_signal_dn,
            )
            if centroid is None:
                observed_centers.append(center)
            else:
                measured.append((centroid[0], centroid[1]))
                qualities.append(float(centroid[2]))
                measured_references.append(center)
                observed_centers.append((centroid[0], centroid[1]))
        if measured:
            quality = np.asarray(qualities, dtype=np.float64)
            keep = quality >= (0.25 * float(np.max(quality)))
            shifts = (
                np.asarray(measured, dtype=np.float64)[keep]
                - np.asarray(measured_references, dtype=np.float64)[keep]
            )
        else:
            # No center has enough signal in this band: report zero shift rather
            # than averaging reference-placeholder centers that were not observed.
            shifts = np.zeros((1, 2), dtype=np.float64)
        estimated_dx = float(np.mean(shifts[:, 0]))
        estimated_dy = float(np.mean(shifts[:, 1]))
        truth_dx = float(truth_shifts[frame][0])
        truth_dy = float(truth_shifts[frame][1])
        shift_norm = float(np.hypot(estimated_dx, estimated_dy))
        max_shift = max(max_shift, shift_norm)
        if shift_norm > config.max_shift_px:
            raise WorkbenchError(
                ErrorCode.E_DRIFT_BIAS,
                "estimated drift exceeds declared tolerance",
                {
                    "frame": frame,
                    "estimated_shift_px": [estimated_dx, estimated_dy],
                    "estimated_shift_norm_px": shift_norm,
                    "tolerance_px": config.max_shift_px,
                },
            )
        residual_dx = estimated_dx - truth_dx
        residual_dy = estimated_dy - truth_dy
        residual_norm = float(np.hypot(residual_dx, residual_dy))
        residuals.append(residual_norm)
        frame_results.append(
            DriftFrameResult(
                frame=frame,
                estimated_dx=estimated_dx,
                estimated_dy=estimated_dy,
                truth_dx=truth_dx,
                truth_dy=truth_dy,
                residual_dx=residual_dx,
                residual_dy=residual_dy,
                residual_norm=residual_norm,
                max_estimated_norm=shift_norm,
                centers_observed=tuple(observed_centers),
            )
        )
    return DriftSequenceResult(
        frames=tuple(frame_results),
        max_residual_px=float(max(residuals)),
        rms_residual_px=float(np.sqrt(np.mean(np.square(residuals)))),
        max_shift_px=max_shift,
        compensation_applied=bool(config.compensate),
    )


def _bilinear_sample(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    x0 = np.floor(sample_x).astype(np.int64)
    y0 = np.floor(sample_y).astype(np.int64)
    fx = sample_x - x0
    fy = sample_y - y0
    x1 = x0 + 1
    y1 = y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    clipped_x0 = np.clip(x0, 0, width - 1)
    clipped_x1 = np.clip(x1, 0, width - 1)
    clipped_y0 = np.clip(y0, 0, height - 1)
    clipped_y1 = np.clip(y1, 0, height - 1)
    value = (
        (1.0 - fy) * (1.0 - fx) * image[clipped_y0, clipped_x0]
        + (1.0 - fy) * fx * image[clipped_y0, clipped_x1]
        + fy * (1.0 - fx) * image[clipped_y1, clipped_x0]
        + fy * fx * image[clipped_y1, clipped_x1]
    )
    value = np.where(inside, value, 0.0)
    return value, inside


def shift_stack_to_reference(
    stack: np.ndarray,
    invalid_mask: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``input(x+dx, y+dy)`` to undo a measured forward shift ``(dx,dy)``."""
    cube = np.asarray(stack)
    invalid = np.asarray(invalid_mask, dtype=np.uint8)
    single_image = cube.ndim == 2
    if single_image:
        cube = cube[None, :, :]
        invalid = invalid[None, :, :]
    if cube.ndim != 3 or invalid.shape != cube.shape:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "shift input must be a band stack with matching mask")
    height, width = cube.shape[1:]
    yy, xx = np.mgrid[0:height, 0:width]
    sample_x = xx.astype(np.float64) + float(dx)
    sample_y = yy.astype(np.float64) + float(dy)
    shifted = np.empty_like(cube, dtype=np.float32)
    invalid_out = np.zeros((height, width), dtype=np.uint8)
    for band in range(cube.shape[0]):
        value, inside = _bilinear_sample(cube[band].astype(np.float64), sample_x, sample_y)
        shifted[band] = value.astype(np.float32)
        invalid_out |= np.logical_not(inside).astype(np.uint8)
    invalid_float, _ = _bilinear_sample(invalid[0].astype(np.float64), sample_x, sample_y)
    invalid_out |= (invalid_float > 0.5).astype(np.uint8)
    shifted[:, invalid_out.astype(bool)] = 0.0
    if single_image:
        return shifted[0], invalid_out
    return shifted, invalid_out
