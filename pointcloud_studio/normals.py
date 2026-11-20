"""Deterministic radius-based normal re-estimation with per-point audit data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ErrorCode, StudioError
from .spatial import VoxelGrid
from .util import point_quantization_bound, sha256_bytes

DEFAULT_MIN_RADIUS = 0.05
DEFAULT_MAX_RADIUS = 2.0


def validate_neighbor_radius(radius: float, minimum: float = DEFAULT_MIN_RADIUS, maximum: float = DEFAULT_MAX_RADIUS) -> float:
    try:
        value = float(radius)
    except (TypeError, ValueError) as exc:
        raise StudioError(
            ErrorCode.E_NEIGHBOR_RADIUS,
            "neighbor radius must be numeric",
            context={"radius": repr(radius), "minimum": minimum, "maximum": maximum},
        ) from exc
    if not np.isfinite(value) or value < minimum or value > maximum:
        raise StudioError(
            ErrorCode.E_NEIGHBOR_RADIUS,
            "neighbor radius is outside the supported range",
            context={"radius": value, "minimum": minimum, "maximum": maximum},
        )
    return value


@dataclass(frozen=True)
class NormalReestimateResult:
    normals: np.ndarray
    confidence: np.ndarray
    neighbor_counts: np.ndarray
    residuals: np.ndarray
    error_bounds: np.ndarray
    radius: float
    selected_indices: np.ndarray
    selection_sha256: str

    def audit(self) -> dict[str, Any]:
        selected = self.selected_indices
        if selected.size:
            counts = self.neighbor_counts[selected]
            residuals = self.residuals[selected]
            bounds = self.error_bounds[selected]
        else:
            counts = np.zeros(0, dtype=np.int32)
            residuals = np.zeros(0, dtype=np.float32)
            bounds = np.zeros(0, dtype=np.float32)
        return {
            "neighbor_radius": self.radius,
            "selected_point_count": int(selected.size),
            "selection_sha256": self.selection_sha256,
            "neighbor_count": {
                "minimum": int(counts.min()) if counts.size else 0,
                "maximum": int(counts.max()) if counts.size else 0,
                "mean": round(float(counts.mean()), 9) if counts.size else 0.0,
            },
            "fit_residual": {
                "minimum": round(float(residuals.min()), 12) if residuals.size else 0.0,
                "maximum": round(float(residuals.max()), 12) if residuals.size else 0.0,
                "mean": round(float(residuals.mean()), 12) if residuals.size else 0.0,
            },
            "normal_error_bound_radians": {
                "minimum": round(float(bounds.min()), 12) if bounds.size else 0.0,
                "maximum": round(float(bounds.max()), 12) if bounds.size else 0.0,
                "mean": round(float(bounds.mean()), 12) if bounds.size else 0.0,
            },
        }


def reestimate_normals_in_region(
    points: np.ndarray,
    normals: np.ndarray,
    confidence: np.ndarray,
    error_bounds: np.ndarray,
    selected_indices: np.ndarray,
    neighbor_radius: float,
    *,
    minimum_radius: float = DEFAULT_MIN_RADIUS,
    maximum_radius: float = DEFAULT_MAX_RADIUS,
    minimum_neighbors: int = 4,
    previous_neighbor_counts: np.ndarray | None = None,
    previous_residuals: np.ndarray | None = None,
) -> NormalReestimateResult:
    """Re-fit only selected points.

    Every unselected normal and confidence value remains byte-for-byte equal to
    its input value.  The surrounding points are only used as fit neighbours.
    """

    radius = validate_neighbor_radius(neighbor_radius, minimum_radius, maximum_radius)
    source_points = np.asarray(points, dtype=np.float64)
    source_normals = np.asarray(normals, dtype=np.float32)
    source_confidence = np.asarray(confidence, dtype=np.float32)
    source_bounds = np.asarray(error_bounds, dtype=np.float32)
    selected = np.unique(np.asarray(selected_indices, dtype=np.int64))
    if selected.size and (selected.min() < 0 or selected.max() >= source_points.shape[0]):
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal re-estimation selection index is out of range",
            context={
                "selected_count": int(selected.size),
                "point_count": int(source_points.shape[0]),
                "first_bad_index": int(selected[(selected < 0) | (selected >= source_points.shape[0])][0]),
            },
        )

    updated_normals = source_normals.copy()
    updated_confidence = source_confidence.copy()
    if previous_neighbor_counts is None:
        neighbor_counts = np.full(source_points.shape[0], -1, dtype=np.int32)
    else:
        neighbor_counts = np.asarray(previous_neighbor_counts, dtype=np.int32).copy()
    if previous_residuals is None:
        residuals = np.full(source_points.shape[0], np.nan, dtype=np.float32)
    else:
        residuals = np.asarray(previous_residuals, dtype=np.float32).copy()
    updated_bounds = source_bounds.copy()
    quantization = point_quantization_bound(source_points)
    grid = VoxelGrid.build(source_points, radius)

    for index in selected:
        local = grid.query_radius(source_points, source_points[index], radius)
        local_delta = source_points[local] - np.mean(source_points[local], axis=0)
        weights = np.maximum(source_confidence[local].astype(np.float64), 0.05)
        weight_sum = float(weights.sum())
        neighbor_counts[index] = int(local.size)
        if local.size < minimum_neighbors or weight_sum <= 0.0:
            updated_confidence[index] = np.float32(0.0)
            residuals[index] = np.float32(radius)
            updated_bounds[index] = np.float32(np.pi / 2.0)
            continue

        weighted = local_delta * np.sqrt(weights[:, None])
        covariance = (weighted.T @ weighted) / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        candidate = eigenvectors[:, order[0]].astype(np.float64)
        if float(np.dot(candidate, source_normals[index].astype(np.float64))) < 0.0:
            candidate = -candidate
        length = float(np.linalg.norm(candidate))
        if length <= np.finfo(np.float64).tiny:
            updated_confidence[index] = np.float32(0.0)
            residuals[index] = np.float32(radius)
            updated_bounds[index] = np.float32(np.pi / 2.0)
            continue
        candidate /= length
        updated_normals[index] = candidate.astype(np.float32)

        # ``covariance`` is already normalised by the total weight.
        residual = float(np.sqrt(max(eigenvalues[0], 0.0)))
        residuals[index] = np.float32(residual)
        condition = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[1] + np.finfo(np.float64).eps)
        coherence = float(np.clip(condition, 0.0, 1.0))
        support = min(1.0, float(local.size) / 24.0)
        sharpness = float(np.exp(-((residual / max(0.08 * radius, 1.0e-12)) ** 2)))
        fitted_confidence = 0.05 + 0.95 * (0.55 * coherence + 0.25 * support + 0.20 * sharpness)
        updated_confidence[index] = np.float32(np.clip(fitted_confidence, 0.0, 1.0))

        angle_bound = float(
            np.arctan2(residual + quantization[index], max(0.5 * radius, 1.0e-12))
            + 0.02 * (1.0 - coherence)
        )
        updated_bounds[index] = np.float32(min(np.pi / 2.0, max(angle_bound, 1.0e-9)))

    selection_payload = np.ascontiguousarray(selected, dtype="<i8").tobytes()
    return NormalReestimateResult(
        normals=updated_normals,
        confidence=updated_confidence,
        neighbor_counts=neighbor_counts,
        residuals=residuals,
        error_bounds=updated_bounds,
        radius=radius,
        selected_indices=selected,
        selection_sha256=sha256_bytes(selection_payload),
    )
