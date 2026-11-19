"""Deterministic floating-point polygon geometry helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .errors import ErrorCode, GlyphError

Vec = tuple[float, float]
Loop = list[Vec]

DEFAULT_SNAP = 1e-9


def snap(value: float, quantum: float = DEFAULT_SNAP) -> float:
    if not math.isfinite(value):
        raise GlyphError(ErrorCode.BOOLEAN_DEGENERATE, "non-finite geometric coordinate")
    rounded = round(value / quantum) * quantum
    if abs(rounded) < quantum * 0.5:
        return 0.0
    return float(rounded)


def snap_point(point: Vec, quantum: float = DEFAULT_SNAP) -> Vec:
    return (snap(point[0], quantum), snap(point[1], quantum))


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec, scalar: float) -> Vec:
    return (a[0] * scalar, a[1] * scalar)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec, b: Vec) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def distance(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: Vec, b: Vec, t: float) -> Vec:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def unit(a: Vec) -> Vec:
    length = norm(a)
    if length == 0.0:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def left_normal(direction: Vec) -> Vec:
    return (-direction[1], direction[0])


def signed_area(loop: Sequence[Vec]) -> float:
    if len(loop) < 3:
        return 0.0
    total = 0.0
    x0, y0 = loop[-1]
    for x1, y1 in loop:
        total += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return total * 0.5


def loop_perimeter(loop: Sequence[Vec]) -> float:
    total = 0.0
    for idx in range(len(loop)):
        total += distance(loop[idx - 1], loop[idx])
    return total


def _on_segment(point: Vec, a: Vec, b: Vec, eps: float) -> bool:
    if abs(cross(sub(b, a), sub(point, a))) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
    )


def winding_number(point: Vec, loops: Sequence[Sequence[Vec]], *, on_boundary_eps: float = 1e-10) -> int:
    """Non-zero winding number. A point on an edge is considered boundary."""

    px, py = point
    winding = 0
    for loop in loops:
        if len(loop) < 2:
            continue
        a = loop[-1]
        for b in loop:
            if _on_segment(point, a, b, on_boundary_eps):
                return 1 if winding == 0 else winding
            if a[1] <= py:
                if b[1] > py:
                    orientation = cross(sub(b, a), sub((px, py), a))
                    if orientation > 0.0:
                        winding += 1
            else:
                if b[1] <= py:
                    orientation = cross(sub(b, a), sub((px, py), a))
                    if orientation < 0.0:
                        winding -= 1
            a = b
    return winding


def inside_nonzero(point: Vec, loops: Sequence[Sequence[Vec]], *, boundary_eps: float = 1e-10) -> bool:
    return winding_number(point, loops, on_boundary_eps=boundary_eps) != 0


def ray_intersections(point: Vec, loops: Sequence[Sequence[Vec]]) -> list[dict[str, float | int]]:
    """Horizontal +x ray intersections with winding direction information."""

    px, py = point
    hits: list[dict[str, float | int]] = []
    for loop_index, loop in enumerate(loops):
        if len(loop) < 2:
            continue
        a = loop[-1]
        for edge_index, b in enumerate(loop):
            if (a[1] > py) != (b[1] > py):
                x = a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x >= px - 1e-12:
                    direction = 1 if b[1] > a[1] else -1
                    hits.append(
                        {
                            "loop": loop_index,
                            "edge": edge_index,
                            "x": x,
                            "direction": direction,
                            "ax": a[0],
                            "ay": a[1],
                            "bx": b[0],
                            "by": b[1],
                        }
                    )
            a = b
    hits.sort(key=lambda hit: (float(hit["x"]), int(hit["loop"]), int(hit["edge"])))
    return hits


def nearest_point_on_segment(point: Vec, a: Vec, b: Vec) -> tuple[Vec, float, float]:
    ab = sub(b, a)
    denom = dot(ab, ab)
    if denom == 0.0:
        return a, distance(point, a), 0.0
    t = max(0.0, min(1.0, dot(sub(point, a), ab) / denom))
    closest = add(a, mul(ab, t))
    return closest, distance(point, closest), t


def directed_hausdorff(
    source: Sequence[Sequence[Vec]], target: Sequence[Sequence[Vec]]
) -> dict[str, float | int | Vec | None]:
    best = -1.0
    details: dict[str, float | int | Vec | None] = {
        "distance": 0.0,
        "source_loop": None,
        "source_edge": None,
        "target_loop": None,
        "target_edge": None,
        "point": None,
    }
    target_edges = [(li, ei, loop[ei - 1], edge) for li, loop in enumerate(target) for ei, edge in enumerate(loop)]
    if not target_edges:
        return details
    for source_loop, loop in enumerate(source):
        for edge_index, point in enumerate(loop):
            local_best = math.inf
            local_index = 0
            for target_index, (_, _, a, b) in enumerate(target_edges):
                dist = nearest_point_on_segment(point, a, b)[1]
                if dist < local_best:
                    local_best = dist
                    local_index = target_index
            if local_best > best:
                target_loop, target_edge, _, _ = target_edges[local_index]
                best = local_best
                details = {
                    "distance": best,
                    "source_loop": source_loop,
                    "source_edge": edge_index,
                    "target_loop": target_loop,
                    "target_edge": target_edge,
                    "point": point,
                }
    return details


def hausdorff_distance(source: Sequence[Sequence[Vec]], target: Sequence[Sequence[Vec]]) -> dict[str, object]:
    forward = directed_hausdorff(source, target)
    backward = directed_hausdorff(target, source)
    if float(forward["distance"]) >= float(backward["distance"]):
        return {"distance": forward["distance"], "direction": "a_to_b", "forward": forward, "backward": backward}
    return {"distance": backward["distance"], "direction": "b_to_a", "forward": forward, "backward": backward}


def segment_intersection_params(a: Vec, b: Vec, c: Vec, d: Vec, *, eps: float = 1e-12) -> list[tuple[float, float]]:
    """Return parameters for proper, endpoint, and collinear-overlap intersections."""

    r = sub(b, a)
    s = sub(d, c)
    r_len = norm(r)
    s_len = norm(s)
    if r_len <= eps:
        if s_len <= eps:
            return [(0.0, 0.0)] if distance(a, c) <= eps else []
        u = dot(sub(a, c), s) / dot(s, s)
        return [(0.0, u)] if -eps <= u <= 1.0 + eps else []
    if s_len <= eps:
        t = dot(sub(c, a), r) / dot(r, r)
        return [(t, 0.0)] if -eps <= t <= 1.0 + eps else []

    denom = cross(r, s)
    qminus_p = sub(c, a)
    scale = max(1.0, r_len * s_len)
    if abs(denom) > eps * scale:
        t = cross(qminus_p, s) / denom
        u = cross(qminus_p, r) / denom
        if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
            return [(min(1.0, max(0.0, t)), min(1.0, max(0.0, u)))]
        return []

    if abs(cross(qminus_p, r)) > eps * max(1.0, r_len * distance(a, c)):
        return []

    rr = dot(r, r)
    ss = dot(s, s)
    pairs: list[tuple[float, float]] = []
    for t in (0.0, 1.0):
        point = add(a, mul(r, t))
        u = dot(sub(point, c), s) / ss
        if -eps <= u <= 1.0 + eps:
            pairs.append((t, min(1.0, max(0.0, u))))
    for u in (0.0, 1.0):
        point = add(c, mul(s, u))
        t = dot(sub(point, a), r) / rr
        if -eps <= t <= 1.0 + eps:
            pairs.append((min(1.0, max(0.0, t)), u))
    return sorted(set(pairs))


def clean_loop(loop: Iterable[Vec], *, quantum: float = DEFAULT_SNAP, min_points: int = 3) -> Loop:
    cleaned: Loop = []
    for raw in loop:
        point = snap_point(raw, quantum)
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < min_points:
        return []
    return cleaned


def merge_collinear(loop: Loop, *, eps: float = 1e-10) -> Loop:
    if len(loop) < 3:
        return list(loop)
    changed = True
    points = list(loop)
    while changed and len(points) >= 3:
        changed = False
        output: Loop = []
        count = len(points)
        for idx, current in enumerate(points):
            previous = output[-1] if output else points[-1]
            following = points[(idx + 1) % count]
            if current == previous or current == following:
                changed = True
                continue
            v1 = sub(current, previous)
            v2 = sub(following, current)
            if abs(cross(v1, v2)) <= eps and dot(v1, v2) >= 0.0:
                changed = True
                continue
            output.append(current)
        if len(output) < 3:
            return []
        points = output
    return points


def normalize_loops(
    loops: Sequence[Sequence[Vec]],
    *,
    quantum: float = DEFAULT_SNAP,
    merge: bool = True,
    orient: bool = True,
) -> list[Loop]:
    cleaned: list[Loop] = []
    for loop in loops:
        candidate = clean_loop(loop, quantum=quantum)
        if merge:
            candidate = merge_collinear(candidate)
        if len(candidate) >= 3 and abs(signed_area(candidate)) > quantum * quantum:
            cleaned.append(candidate)

    if orient:
        for idx, loop in enumerate(cleaned):
            probe = _interior_probe(loop, quantum)
            depth = sum(1 for j, other in enumerate(cleaned) if j != idx and inside_nonzero(probe, [other]))
            want_ccw = depth % 2 == 0
            if (signed_area(loop) > 0.0) != want_ccw:
                loop.reverse()

    canonical: list[Loop] = []
    for loop in cleaned:
        start = min(range(len(loop)), key=lambda i: loop[i])
        canonical.append(loop[start:] + loop[:start])
    canonical.sort(key=lambda loop: tuple(loop))
    return canonical


def _interior_probe(loop: Loop, quantum: float) -> Vec:
    # A short offset to the left of the first non-degenerate edge avoids exact
    # shared-vertex tests in the normal case.
    for idx in range(len(loop)):
        a = loop[idx]
        b = loop[(idx + 1) % len(loop)]
        direction = unit(sub(b, a))
        if direction == (0.0, 0.0):
            continue
        normal = left_normal(direction)
        offset = max(quantum * 8.0, 1e-8)
        return add(mul(add(a, b), 0.5), mul(normal, offset))
    return loop[0]


def bounds(loops: Sequence[Sequence[Vec]]) -> tuple[float, float, float, float]:
    points = [point for loop in loops for point in loop]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    )


def total_abs_area(loops: Sequence[Sequence[Vec]]) -> float:
    return sum(abs(signed_area(loop)) for loop in loops)


def clip_signed_area_to_box(points: Sequence[Vec], x0: float, y0: float, x1: float, y1: float) -> float:
    """Sutherland-Hodgman clipping against an axis-aligned pixel box."""

    polygon: list[Vec] = list(points)
    if not polygon:
        return 0.0

    def clip_axis(poly: list[Vec], axis: int, value: float, keep_greater: bool) -> list[Vec]:
        if not poly:
            return []
        output: list[Vec] = []
        previous = poly[-1]
        previous_inside = previous[axis] >= value if keep_greater else previous[axis] <= value
        for current in poly:
            current_inside = current[axis] >= value if keep_greater else current[axis] <= value
            if current_inside != previous_inside:
                denom = current[axis] - previous[axis]
                if denom != 0.0:
                    t = (value - previous[axis]) / denom
                    intersection = (
                        (value, previous[1] + (current[1] - previous[1]) * t)
                        if axis == 0
                        else (previous[0] + (current[0] - previous[0]) * t, value)
                    )
                    output.append(intersection)
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        return output

    polygon = clip_axis(polygon, 0, x0, True)
    polygon = clip_axis(polygon, 0, x1, False)
    polygon = clip_axis(polygon, 1, y0, True)
    polygon = clip_axis(polygon, 1, y1, False)
    if len(polygon) < 3:
        return 0.0
    return signed_area(polygon)
