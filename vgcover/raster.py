"""Pixel-centre horizontal-ray coverage for nonzero and even-odd rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import CoverageError, ErrorCode
from .scene import VALID_FILL_RULES, validate_fill_rule


RASTER_POLICY = {
    "version": "vgcover.raster.pixel_centre_horizontal_ray.v1",
    "sample": "pixel centre (x + 0.5, y + 0.5)",
    "ray_direction": "+x",
    "y_edge_test": "half-open [min_y, max_y)",
    "x_boundary_test": "intersection_x > sample_x (strict)",
    "coverage_dtype": "float32",
    "dependency_metric": "number of rightward polyline-edge intersections used by the pixel",
}


@dataclass
class CoverageMap:
    fill_rule: str
    coverage: np.ndarray
    dependency_edge_counts: np.ndarray
    winding_counts: np.ndarray
    edge_count: int
    width: int
    height: int
    policy: dict[str, Any]


@dataclass
class DualCoverage:
    nonzero: CoverageMap
    evenodd: CoverageMap
    difference: np.ndarray
    difference_pixel_count: int
    edge_count: int
    width: int
    height: int

    def selected(self, fill_rule: str) -> CoverageMap:
        validate_fill_rule(fill_rule)
        return self.nonzero if fill_rule == "nonzero" else self.evenodd


def _normalise_polylines(polylines: Iterable[Sequence[Sequence[float]]]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for contour_index, polyline in enumerate(polylines):
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "polyline must be an N x 2 array with at least two vertices",
                contour_index=contour_index,
                details={"shape": list(points.shape)},
            )
        if not np.all(np.isfinite(points)):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "polyline contains non-finite coordinates",
                contour_index=contour_index,
            )
        result.append(points)
    return result


def _edge_arrays(polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0_chunks = []
    y0_chunks = []
    x1_chunks = []
    y1_chunks = []
    for points in polylines:
        x0_chunks.append(points[:, 0])
        y0_chunks.append(points[:, 1])
        x1_chunks.append(np.roll(points[:, 0], -1))
        y1_chunks.append(np.roll(points[:, 1], -1))
    if not x0_chunks:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    return (
        np.concatenate(x0_chunks).astype(np.float64, copy=False),
        np.concatenate(y0_chunks).astype(np.float64, copy=False),
        np.concatenate(x1_chunks).astype(np.float64, copy=False),
        np.concatenate(y1_chunks).astype(np.float64, copy=False),
    )


def _rasterize_core(
    polylines: list[np.ndarray], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise CoverageError(ErrorCode.SCENE_INVALID, "width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise CoverageError(ErrorCode.SCENE_INVALID, "height must be a positive integer")

    x0, y0, x1, y1 = _edge_arrays(polylines)
    edge_count = int(x0.size)
    dependency = np.zeros((height, width), dtype=np.int32)
    winding = np.zeros((height, width), dtype=np.int32)
    if edge_count == 0:
        return np.zeros((height, width), dtype=np.float32), dependency, winding

    y_centres = np.arange(height, dtype=np.float64) + 0.5
    x_centres = np.arange(width, dtype=np.float64) + 0.5
    dy = y1 - y0
    valid_edges = dy != 0.0
    edge_increment = np.where(y1 > y0, 1, -1).astype(np.int32)

    for row, sample_y in enumerate(y_centres):
        lower = np.minimum(y0, y1)
        upper = np.maximum(y0, y1)
        active = valid_edges & (sample_y >= lower) & (sample_y < upper)
        if not np.any(active):
            continue
        active_indices = np.nonzero(active)[0]
        ax0 = x0[active_indices]
        ay0 = y0[active_indices]
        ady = dy[active_indices]
        adx = x1[active_indices] - ax0
        intersections = ax0 + (sample_y - ay0) * adx / ady
        rightward = intersections[None, :] > x_centres[:, None]
        counts = np.sum(rightward, axis=1, dtype=np.int32)
        increments = edge_increment[active_indices]
        row_windings = rightward.astype(np.int32) @ increments
        dependency[row, :] = counts
        winding[row, :] = row_windings.astype(np.int32)

    return np.zeros((height, width), dtype=np.float32), dependency, winding


def rasterize_polylines_dual(
    polylines: Iterable[Sequence[Sequence[float]]],
    width: int,
    height: int,
) -> DualCoverage:
    """Rasterize once and return both public fill rules."""

    normalised = _normalise_polylines(polylines)
    _, dependency, winding = _rasterize_core(normalised, width, height)
    nonzero_coverage = (winding != 0).astype(np.float32)
    evenodd_coverage = ((dependency & 1) != 0).astype(np.float32)
    difference = (nonzero_coverage - evenodd_coverage).astype(np.float32)
    edge_count = sum(int(points.shape[0]) for points in normalised)

    nonzero = CoverageMap(
        fill_rule="nonzero",
        coverage=nonzero_coverage,
        dependency_edge_counts=dependency,
        winding_counts=winding,
        edge_count=edge_count,
        width=width,
        height=height,
        policy=RASTER_POLICY,
    )
    evenodd = CoverageMap(
        fill_rule="evenodd",
        coverage=evenodd_coverage,
        dependency_edge_counts=dependency,
        winding_counts=winding,
        edge_count=edge_count,
        width=width,
        height=height,
        policy=RASTER_POLICY,
    )
    return DualCoverage(
        nonzero=nonzero,
        evenodd=evenodd,
        difference=difference,
        difference_pixel_count=int(np.count_nonzero(difference)),
        edge_count=edge_count,
        width=width,
        height=height,
    )


def rasterize_polylines(
    polylines: Iterable[Sequence[Sequence[float]]],
    width: int,
    height: int,
    fill_rule: str,
) -> CoverageMap:
    validate_fill_rule(fill_rule)
    dual = rasterize_polylines_dual(polylines, width, height)
    return dual.selected(fill_rule)


def rasterize_geometry_dual(
    geometry: dict[str, Any], width: int, height: int
) -> DualCoverage:
    polylines = [np.asarray(contour["points"], dtype=np.float64) for contour in geometry["contours"]]
    return rasterize_polylines_dual(polylines, width, height)


def rasterize_geometry(
    geometry: dict[str, Any], width: int, height: int, fill_rule: str
) -> CoverageMap:
    validate_fill_rule(
        fill_rule,
        region_id=int(geometry["region_id"]) if geometry.get("region_id") is not None else None,
    )
    dual = rasterize_geometry_dual(geometry, width, height)
    return dual.selected(fill_rule)


def coverage_difference(
    coverage_a: np.ndarray, coverage_b: np.ndarray
) -> np.ndarray:
    """Return a per-pixel float32 difference without changing dtype."""

    left = np.asarray(coverage_a, dtype=np.float32)
    right = np.asarray(coverage_b, dtype=np.float32)
    if left.shape != right.shape:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "coverage maps must have identical shapes for difference export",
            details={"left_shape": list(left.shape), "right_shape": list(right.shape)},
        )
    return (left - right).astype(np.float32)
