from __future__ import annotations

import copy
import math
from typing import Any

from .canonical import sha256_value
from .scene import _is_degenerate_segment

MAX_DEPTH = 20


def _distance_point_to_line(point: list[float], start: list[float], end: list[float]) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length <= 1e-15:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(vx * (point[1] - start[1]) - vy * (point[0] - start[0])) / length


def _flatness(p0: list[float], c1: list[float], c2: list[float], p1: list[float]) -> float:
    return max(_distance_point_to_line(c1, p0, p1), _distance_point_to_line(c2, p0, p1))


def _middle(a: list[float], b: list[float]) -> list[float]:
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]


def _split_cubic(
    p0: list[float], c1: list[float], c2: list[float], p1: list[float]
) -> tuple[tuple[list[float], list[float], list[float], list[float]], tuple[list[float], list[float], list[float], list[float]]]:
    a = _middle(p0, c1)
    b = _middle(c1, c2)
    c = _middle(c2, p1)
    d = _middle(a, b)
    e = _middle(b, c)
    f = _middle(d, e)
    return (p0, a, d, f), (f, e, c, p1)


def _flatten_cubic(
    p0: list[float], c1: list[float], c2: list[float], p1: list[float], tolerance: float
) -> tuple[list[list[float]], float]:
    points: list[list[float]] = []
    max_error = 0.0

    def visit(a: list[float], b: list[float], c: list[float], d: list[float], depth: int) -> None:
        nonlocal max_error
        flatness = _flatness(a, b, c, d)
        if flatness <= tolerance or depth >= MAX_DEPTH:
            if not points:
                points.append([round(a[0], 9), round(a[1], 9)])
            points.append([round(d[0], 9), round(d[1], 9)])
            max_error = max(max_error, flatness)
            return
        left, right = _split_cubic(a, b, c, d)
        visit(*left, depth + 1)
        visit(*right, depth + 1)

    visit(p0, c1, c2, p1, 0)
    return points, round(max_error, 12)


def flatten_segment(segment: dict[str, Any], tolerance: float) -> dict[str, Any]:
    segment_type = segment.get("type")
    if segment_type == "line":
        p0 = [float(value) for value in segment["p0"]]
        p1 = [float(value) for value in segment["p1"]]
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= 1e-12:
            return {
                "points": [],
                "error_estimate": 0.0,
                "vertex_count": 0,
                "action": "skip:zero-length-line",
                "error_method": "exact",
            }
        return {
            "points": [[round(p0[0], 9), round(p0[1], 9)], [round(p1[0], 9), round(p1[1], 9)]],
            "error_estimate": 0.0,
            "vertex_count": 2,
            "action": "emit:line",
            "error_method": "exact",
        }
    if segment_type != "cubic":
        return {
            "points": [],
            "error_estimate": 0.0,
            "vertex_count": 0,
            "action": "skip:unknown-type",
            "error_method": "none",
        }

    p0 = [float(value) for value in segment["p0"]]
    c1 = [float(value) for value in segment["c1"]]
    c2 = [float(value) for value in segment["c2"]]
    p1 = [float(value) for value in segment["p1"]]
    degenerate, reason = _is_degenerate_segment(segment)
    if reason == "zero-length" or reason == "missing-points":
        return {
            "points": [],
            "error_estimate": 0.0,
            "vertex_count": 0,
            "action": "skip:zero-length-cubic",
            "error_method": "exact",
            "degenerate_reason": reason,
        }
    if reason == "collinear-controls":
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length <= 1e-12:
            return {
                "points": [],
                "error_estimate": 0.0,
                "vertex_count": 0,
                "action": "skip:collapsed-collinear-cubic",
                "error_method": "exact",
                "degenerate_reason": reason,
            }
        return {
            "points": [[round(p0[0], 9), round(p0[1], 9)], [round(p1[0], 9), round(p1[1], 9)]],
            "error_estimate": 0.0,
            "vertex_count": 2,
            "action": "degrade:collinear-cubic-to-line",
            "error_method": "exact-after-degrade",
            "degenerate_reason": reason,
        }

    points, error = _flatten_cubic(p0, c1, c2, p1, tolerance)
    return {
        "points": points,
        "error_estimate": error,
        "vertex_count": len(points),
        "action": "subdivide:adaptive",
        "error_method": "control-point-to-chord-bound",
    }


def flatten_subpath(subpath: dict[str, Any], tolerance: float, subpath_index: int = 0) -> dict[str, Any]:
    polyline: list[list[float]] = []
    segment_reports: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(subpath.get("segments", [])):
        report = flatten_segment(segment, tolerance)
        report.update(
            {
                "segment_index": segment_index,
                "segment_type": segment.get("type"),
            }
        )
        segment_reports.append(report)
        for point in report["points"]:
            if not polyline or abs(polyline[-1][0] - point[0]) > 1e-12 or abs(polyline[-1][1] - point[1]) > 1e-12:
                polyline.append(point)
    if subpath.get("closed") and len(polyline) > 1:
        if abs(polyline[-1][0] - polyline[0][0]) <= 1e-12 and abs(polyline[-1][1] - polyline[0][1]) <= 1e-12:
            polyline.pop()
    result = {
        "subpath_index": subpath_index,
        "subpath_id": subpath.get("subpath_id", f"s{subpath_index}"),
        "closed": bool(subpath.get("closed")),
        "tolerance": tolerance,
        "segments": segment_reports,
        "polyline": polyline,
        "vertex_count": len(polyline),
        "max_error_estimate": max((item["error_estimate"] for item in segment_reports), default=0.0),
        "skipped_segment_count": sum(item["action"].startswith("skip:") for item in segment_reports),
        "degraded_segment_count": sum(item["action"].startswith("degrade:") for item in segment_reports),
    }
    return result


def flatten_path(path: dict[str, Any], tolerance: float) -> dict[str, Any]:
    subpaths = [flatten_subpath(subpath, tolerance, index) for index, subpath in enumerate(path.get("subpaths", []))]
    result = {
        "path_id": path.get("path_id"),
        "fill_rule": path.get("fill_rule"),
        "closed": bool(path.get("subpaths")) and all(bool(item.get("closed")) for item in path.get("subpaths", [])),
        "tolerance": tolerance,
        "subpaths": subpaths,
        "vertex_count": sum(item["vertex_count"] for item in subpaths),
        "max_error_estimate": max((item["max_error_estimate"] for item in subpaths), default=0.0),
        "skipped_segment_count": sum(item["skipped_segment_count"] for item in subpaths),
        "degraded_segment_count": sum(item["degraded_segment_count"] for item in subpaths),
    }
    result["flatten_hash"] = sha256_value(result)
    return result


def flatten_scene(scene: dict[str, Any], tolerance: float) -> dict[str, Any]:
    paths = [flatten_path(path, tolerance) for path in scene.get("paths", [])]
    result = {
        "scene_id": scene.get("scene_id"),
        "tolerance": tolerance,
        "paths": paths,
        "path_count": len(paths),
        "vertex_count": sum(path["vertex_count"] for path in paths),
        "max_error_estimate": max((path["max_error_estimate"] for path in paths), default=0.0),
        "skipped_segment_count": sum(path["skipped_segment_count"] for path in paths),
        "degraded_segment_count": sum(path["degraded_segment_count"] for path in paths),
    }
    result["flatten_hash"] = sha256_value(result)
    return result


def polyline_for_subpath(flat_subpath: dict[str, Any]) -> list[list[float]]:
    return copy.deepcopy(flat_subpath["polyline"])
