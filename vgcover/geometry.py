"""Deterministic contour flattening with analytic and measured error metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .errors import CoverageError, ErrorCode
from .hashing import sha256_json
from .scene import validate_region, validate_tolerance


GEOMETRY_POLICY = {
    "version": "vgcover.geometry.cubic_adaptive.v1",
    "line": "one chord; zero geometric error",
    "cubic": "dyadic de Casteljau subdivision; leaf accepted when the maximum distance from its interior control points to its chord segment is <= tolerance",
    "analytic_error_unit": "pixel",
    "measured_error": "maximum distance of quarter/mid/three-quarter curve samples to each leaf chord segment",
    "subdivision_split": "t=(t0+t1)/2",
    "coordinate_rounding_error_bound_px": 1e-12,
    "max_depth": 24,
}


Point = tuple[float, float]


@dataclass(frozen=True)
class CubicLeaf:
    t0: float
    t1: float
    p0: Point
    p1: Point
    p2: Point
    p3: Point
    analytic_bound: float
    measured_chord_height: float
    depth: int


def _distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-30:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * dx + (py - ay) * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = ax + projection * dx
    closest_y = ay + projection * dy
    return math.hypot(px - closest_x, py - closest_y)


def _split_cubic(points: tuple[Point, Point, Point, Point]) -> tuple[
    tuple[Point, Point, Point, Point], tuple[Point, Point, Point, Point]
]:
    p0, p1, p2, p3 = points
    p01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    p12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    p23 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
    p012 = ((p01[0] + p12[0]) * 0.5, (p01[1] + p12[1]) * 0.5)
    p123 = ((p12[0] + p23[0]) * 0.5, (p12[1] + p23[1]) * 0.5)
    mid = ((p012[0] + p123[0]) * 0.5, (p012[1] + p123[1]) * 0.5)
    return (p0, p01, p012, mid), (mid, p123, p23, p3)


def _evaluate_cubic(points: tuple[Point, Point, Point, Point], t: float) -> Point:
    u = 1.0 - t
    p0, p1, p2, p3 = points
    x = (
        u * u * u * p0[0]
        + 3.0 * u * u * t * p1[0]
        + 3.0 * u * t * t * p2[0]
        + t * t * t * p3[0]
    )
    y = (
        u * u * u * p0[1]
        + 3.0 * u * u * t * p1[1]
        + 3.0 * u * t * t * p2[1]
        + t * t * t * p3[1]
    )
    return (x, y)


def _cubic_leaf(points: tuple[Point, Point, Point, Point], t0: float, t1: float, depth: int) -> CubicLeaf:
    p0, p1, p2, p3 = points
    chord_bound = max(
        _distance_point_to_segment(p1, p0, p3),
        _distance_point_to_segment(p2, p0, p3),
    )
    measured = 0.0
    for fraction in (0.25, 0.5, 0.75):
        t = t0 + (t1 - t0) * fraction
        sample = _evaluate_cubic(points, fraction)
        measured = max(measured, _distance_point_to_segment(sample, p0, p3))
    return CubicLeaf(
        t0=t0,
        t1=t1,
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        analytic_bound=chord_bound,
        measured_chord_height=measured,
        depth=depth,
    )


def _flatten_cubic(
    points: tuple[Point, Point, Point, Point], tolerance: float
) -> list[CubicLeaf]:
    max_depth = int(GEOMETRY_POLICY["max_depth"])
    leaves: list[CubicLeaf] = []

    def visit(
        current: tuple[Point, Point, Point, Point],
        t0: float,
        t1: float,
        depth: int,
    ) -> None:
        leaf = _cubic_leaf(current, t0, t1, depth)
        if leaf.analytic_bound <= tolerance or depth >= max_depth:
            leaves.append(leaf)
            return
        left, right = _split_cubic(current)
        midpoint = (t0 + t1) * 0.5
        visit(left, t0, midpoint, depth + 1)
        visit(right, midpoint, t1, depth + 1)

    visit(points, 0.0, 1.0, 0)
    return leaves


def _as_point(value: Any) -> Point:
    return (float(value[0]), float(value[1]))


def _perimeter(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += math.hypot(next_point[0] - point[0], next_point[1] - point[1])
    return total


def _round_for_storage(value: float) -> float:
    """Round serialized geometry without changing the cached numeric map."""

    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def discretize_region(region: dict[str, Any]) -> dict[str, Any]:
    """Flatten every segment in a region and return error diagnostics."""

    validate_region(region)
    tolerance = float(region["tolerance"])
    region_id = int(region["region_id"])
    contour_results: list[dict[str, Any]] = []
    max_analytic = 0.0
    max_measured = 0.0
    total_perimeter = 0.0
    total_vertices = 0
    total_segments = 0
    depth_limited_segments = 0

    for contour_list_index, contour in enumerate(region["contours"]):
        contour_index = int(contour.get("contour_index", contour_list_index))
        segments = contour["segments"]
        polyline_points: list[Point] = []
        segment_results: list[dict[str, Any]] = []
        contour_max_analytic = 0.0
        contour_max_measured = 0.0
        contour_depth_limited = 0

        for segment_list_index, segment in enumerate(segments):
            segment_index = int(segment.get("segment_index", segment_list_index))
            control_points = [_as_point(point) for point in segment["control_points"]]
            if not polyline_points:
                polyline_points.append(control_points[0])

            if segment["type"] == "line":
                leaves = [
                    CubicLeaf(
                        t0=0.0,
                        t1=1.0,
                        p0=control_points[0],
                        p1=control_points[0],
                        p2=control_points[-1],
                        p3=control_points[-1],
                        analytic_bound=0.0,
                        measured_chord_height=0.0,
                        depth=0,
                    )
                ]
                segment_type = "line"
            else:
                cubic_points = (
                    control_points[0],
                    control_points[1],
                    control_points[2],
                    control_points[3],
                )
                leaves = _flatten_cubic(cubic_points, tolerance)
                segment_type = "cubic"

            if polyline_points and not _same_point(polyline_points[-1], leaves[0].p0):
                raise CoverageError(
                    ErrorCode.CONTOUR_UNCLOSED,
                    "internal discretization chain continuity failure",
                    region_id=region_id,
                    contour_index=contour_index,
                    segment_index=segment_index,
                )

            endpoint_points = []
            for leaf in leaves:
                endpoint_points.append(leaf.p3)
            if len(endpoint_points) > 1:
                polyline_points.extend(endpoint_points)
            elif endpoint_points:
                polyline_points.append(endpoint_points[-1])

            analytic_bound = max((leaf.analytic_bound for leaf in leaves), default=0.0)
            measured = max((leaf.measured_chord_height for leaf in leaves), default=0.0)
            depth_limited = sum(
                1 for leaf in leaves if leaf.depth >= int(GEOMETRY_POLICY["max_depth"])
            )
            if depth_limited:
                contour_depth_limited += 1
                depth_limited_segments += 1
            contour_max_analytic = max(contour_max_analytic, analytic_bound)
            contour_max_measured = max(contour_max_measured, measured)
            segment_results.append(
                {
                    "segment_index": segment_index,
                    "type": segment_type,
                    "subdivision_leaf_count": len(leaves),
                    "polyline_vertex_count": len(leaves) + 1,
                    "analytic_error_bound": _round_for_storage(analytic_bound),
                    "measured_chord_height": _round_for_storage(measured),
                    "max_subdivision_depth": max((leaf.depth for leaf in leaves), default=0),
                    "depth_limited_leaf_count": depth_limited,
                    "leaf_parameter_intervals": [
                        [_round_for_storage(leaf.t0), _round_for_storage(leaf.t1)]
                        for leaf in leaves
                    ],
                }
            )
            total_segments += 1

        if len(polyline_points) > 1 and _same_point(polyline_points[0], polyline_points[-1]):
            polyline_points.pop()

        contour_perimeter = _perimeter(polyline_points)
        total_perimeter += contour_perimeter
        total_vertices += len(polyline_points)
        max_analytic = max(max_analytic, contour_max_analytic)
        max_measured = max(max_measured, contour_max_measured)
        contour_results.append(
            {
                "contour_index": contour_index,
                "orientation": contour["orientation"],
                "closed": True,
                "points": [
                    [_round_for_storage(point[0]), _round_for_storage(point[1])]
                    for point in polyline_points
                ],
                "vertex_count": len(polyline_points),
                "perimeter_px": _round_for_storage(contour_perimeter),
                "max_analytic_error_bound_px": _round_for_storage(contour_max_analytic),
                "max_measured_chord_height_px": _round_for_storage(contour_max_measured),
                "depth_limited_segment_count": contour_depth_limited,
                "segments": segment_results,
            }
        )

    geometry = {
        "policy": GEOMETRY_POLICY,
        "region_id": region_id,
        "tolerance": tolerance,
        "coordinate_space": "pixel",
        "contours": contour_results,
        "summary": {
            "contour_count": len(contour_results),
            "segment_count": total_segments,
            "polyline_vertex_count": total_vertices,
            "perimeter_px": _round_for_storage(total_perimeter),
            "max_analytic_error_bound_px": _round_for_storage(max_analytic),
            "max_measured_chord_height_px": _round_for_storage(max_measured),
            "depth_limited_segment_count": depth_limited_segments,
        },
    }
    geometry["geometry_hash"] = sha256_json(geometry)
    return geometry


def discretize_contour(
    contour: dict[str, Any],
    tolerance: float,
    *,
    region_id: int = 0,
    contour_index: int | None = None,
) -> dict[str, Any]:
    """Convenience API for flattening one contour in isolation.

    The returned object is the same contour report embedded in
    :func:`discretize_region`, making single-contour tests and integrations
    straightforward without weakening the closed-scene validation path.
    """

    tolerance_value = validate_tolerance(tolerance, region_id=region_id)
    contour_copy = dict(contour)
    if contour_index is not None:
        contour_copy["contour_index"] = int(contour_index)
    wrapper = {
        "region_id": int(region_id),
        "category": "single_contour",
        "fill_rule": "nonzero",
        "tolerance": tolerance_value,
        "fill_color": [0, 0, 0],
        "alpha": 1.0,
        "contours": [contour_copy],
        "degenerate": {
            "flagged": False,
            "reason": None,
            "minimum_feature_width_px": None,
            "threshold_px": 0.5,
            "retained": True,
        },
    }
    return discretize_region(wrapper)["contours"][0]


def _same_point(a: Point, b: Point, epsilon: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= epsilon and abs(a[1] - b[1]) <= epsilon


def contour_polylines(geometry: dict[str, Any]) -> list[np.ndarray]:
    """Convert a geometry report to contiguous float64 point arrays."""

    polylines = []
    for contour in geometry["contours"]:
        points = np.asarray(contour["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "cannot rasterize a contour with fewer than two vertices",
                region_id=int(geometry.get("region_id")) if geometry.get("region_id") is not None else None,
                contour_index=int(contour.get("contour_index")) if contour.get("contour_index") is not None else None,
            )
        polylines.append(points)
    return polylines


def geometry_vertex_count(geometry: dict[str, Any]) -> int:
    return int(geometry["summary"]["polyline_vertex_count"])


def geometry_point_sequence(geometry: dict[str, Any]) -> list[list[list[float]]]:
    return [contour["points"] for contour in geometry["contours"]]
