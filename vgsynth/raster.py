from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .canonical import sha256_value

SUBSCANLINES = 8
STROKE_CIRCLE_SIDES = 12
EPSILON = 1e-12


def _clean_polyline(points: list[list[float]], closed: bool) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if not cleaned or abs(cleaned[-1][0] - x) > EPSILON or abs(cleaned[-1][1] - y) > EPSILON:
            cleaned.append([x, y])
    if closed and len(cleaned) > 1:
        if abs(cleaned[-1][0] - cleaned[0][0]) <= EPSILON and abs(cleaned[-1][1] - cleaned[0][1]) <= EPSILON:
            cleaned.pop()
    return cleaned


def _signed_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return total * 0.5


def _positive_polygon(points: list[list[float]]) -> list[list[float]]:
    return list(reversed(points)) if _signed_area(points) < 0.0 else points


def _add_span(row: np.ndarray, start: float, end: float) -> None:
    width = row.shape[0]
    start = max(0.0, float(start))
    end = min(float(width), float(end))
    if end <= start:
        return
    first = int(math.floor(start))
    last = int(math.ceil(end)) - 1
    for column in range(first, last + 1):
        overlap = min(end, column + 1.0) - max(start, float(column))
        if overlap > 0.0:
            row[column] += overlap


def _edge_crossings(edges: list[tuple[float, float, float, float]], y: float) -> list[tuple[float, int]]:
    crossings: list[tuple[float, int]] = []
    for x0, y0, x1, y1 in edges:
        if y0 == y1:
            continue
        if not ((y0 <= y < y1) or (y1 <= y < y0)):
            continue
        t = (y - y0) / (y1 - y0)
        x = x0 + t * (x1 - x0)
        delta = 1 if y1 > y0 else -1
        crossings.append((x, delta))
    crossings.sort(key=lambda item: (item[0], item[1]))
    return crossings


def _group_crossings(crossings: list[tuple[float, int]]) -> list[tuple[float, int, int]]:
    groups: list[list[float | int]] = []
    for x, delta in crossings:
        if groups and abs(float(groups[-1][0]) - x) <= 1e-10:
            groups[-1][1] = int(groups[-1][1]) + delta
            groups[-1][2] = int(groups[-1][2]) + 1
        else:
            groups.append([x, delta, 1])
    return [(float(x), int(delta), int(count)) for x, delta, count in groups]


def _scanline_spans(edges: list[tuple[float, float, float, float]], y: float, rule: str) -> list[tuple[float, float]]:
    groups = _group_crossings(_edge_crossings(edges, y))
    spans: list[tuple[float, float]] = []
    winding = 0
    parity = 0
    previous_x: float | None = None
    active = False
    for x, delta, count in groups:
        if previous_x is not None and active and x > previous_x:
            spans.append((previous_x, x))
        if rule == "nonzero":
            winding += delta
            active = winding != 0
        else:
            parity = (parity + count) & 1
            active = parity == 1
        previous_x = x
    return spans


def rasterize_polygons(
    polygons: list[list[list[float]]],
    width: int,
    height: int,
    rule: str,
    subscanlines: int = SUBSCANLINES,
) -> np.ndarray:
    edges: list[tuple[float, float, float, float]] = []
    for polygon in polygons:
        points = _clean_polyline(polygon, True)
        if len(points) < 3:
            continue
        for index, point in enumerate(points):
            nxt = points[(index + 1) % len(points)]
            if abs(point[1] - nxt[1]) > EPSILON:
                edges.append((point[0], point[1], nxt[0], nxt[1]))
    coverage = np.zeros((height, width), dtype=np.float64)
    if not edges:
        return coverage.astype(np.uint8)
    for row_index in range(height):
        row = coverage[row_index]
        for sample_index in range(subscanlines):
            y = row_index + (sample_index + 0.5) / subscanlines
            for start, end in _scanline_spans(edges, y, rule):
                _add_span(row, start, end)
        row /= float(subscanlines)
    quantized = np.floor(np.clip(coverage, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return quantized


def path_fill_polygons(flat_path: dict[str, Any]) -> list[list[list[float]]]:
    polygons = []
    for subpath in flat_path.get("subpaths", []):
        if not subpath.get("closed"):
            continue
        points = _clean_polyline(subpath.get("polyline", []), True)
        if len(points) >= 3 and abs(_signed_area(points)) > EPSILON:
            polygons.append(points)
    return polygons


def coverage_info(coverage: np.ndarray) -> dict[str, Any]:
    raw = coverage.tobytes()
    return {
        "width": int(coverage.shape[1]),
        "height": int(coverage.shape[0]),
        "content_hash": hashlib.sha256(raw).hexdigest(),
        "covered_pixels": int(np.count_nonzero(coverage)),
        "coverage_sum": int(coverage.astype(np.uint64).sum()),
    }


def first_difference(nonzero: np.ndarray, evenodd: np.ndarray) -> dict[str, Any] | None:
    difference = nonzero != evenodd
    count = int(np.count_nonzero(difference))
    if count == 0:
        return None
    flat_index = int(np.flatnonzero(difference.reshape(-1))[0])
    y, x = divmod(flat_index, nonzero.shape[1])
    return {
        "count": count,
        "x": int(x),
        "y": int(y),
        "nonzero": int(nonzero[y, x]),
        "evenodd": int(evenodd[y, x]),
    }


def rasterize_fill(flat_path: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    polygons = path_fill_polygons(flat_path)
    nonzero = rasterize_polygons(polygons, width, height, "nonzero")
    evenodd = rasterize_polygons(polygons, width, height, "evenodd")
    selected = flat_path.get("fill_rule")
    if selected not in ("nonzero", "evenodd"):
        selected = "nonzero"
    return {
        "selected_rule": selected,
        "nonzero": nonzero,
        "evenodd": evenodd,
        "nonzero_info": coverage_info(nonzero),
        "evenodd_info": coverage_info(evenodd),
        "first_difference": first_difference(nonzero, evenodd),
        "polygon_count": len(polygons),
    }


def _circle_polygon(cx: float, cy: float, radius: float, sides: int = STROKE_CIRCLE_SIDES) -> list[list[float]]:
    points = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        points.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    return _positive_polygon(points)


def _segment_quad(p0: list[float], p1: list[float], half_width: float) -> list[list[float]] | None:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return None
    nx, ny = -dy / length * half_width, dx / length * half_width
    return _positive_polygon(
        [
            [p0[0] + nx, p0[1] + ny],
            [p1[0] + nx, p1[1] + ny],
            [p1[0] - nx, p1[1] - ny],
            [p0[0] - nx, p0[1] - ny],
        ]
    )


def build_stroke_outline(flat_path: dict[str, Any], stroke_width: float) -> list[list[list[float]]]:
    if stroke_width <= 0.0:
        return []
    half_width = stroke_width * 0.5
    polygons: list[list[list[float]]] = []
    for subpath in flat_path.get("subpaths", []):
        points = _clean_polyline(subpath.get("polyline", []), bool(subpath.get("closed")))
        if not points:
            continue
        if len(points) == 1:
            polygons.append(_circle_polygon(points[0][0], points[0][1], half_width))
            continue
        closed = bool(subpath.get("closed"))
        for index in range(len(points) - (0 if closed else 1)):
            p0 = points[index]
            p1 = points[(index + 1) % len(points)]
            quad = _segment_quad(p0, p1, half_width)
            if quad is not None:
                polygons.append(quad)
        for point in points:
            polygons.append(_circle_polygon(point[0], point[1], half_width))
    return polygons


def rasterize_stroke(flat_path: dict[str, Any], width: int, height: int, stroke_width: float) -> dict[str, Any]:
    polygons = build_stroke_outline(flat_path, stroke_width)
    coverage = rasterize_polygons(polygons, width, height, "nonzero")
    outline_hash = sha256_value(
        {
            "stroke_width": stroke_width,
            "circle_sides": STROKE_CIRCLE_SIDES,
            "polygons": [
                [[round(float(point[0]), 9), round(float(point[1]), 9)] for point in polygon]
                for polygon in polygons
            ],
        }
    )
    return {
        "coverage": coverage,
        "info": coverage_info(coverage),
        "outline_hash": outline_hash,
        "polygon_count": len(polygons),
        "vertex_count": sum(len(polygon) for polygon in polygons),
    }
