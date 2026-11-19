"""Open-path stroke expansion with deterministic caps, joins, and arc error."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from .boolean import BooleanOp, BooleanResult, ResolutionOrder, boolean_operation
from .errors import ErrorCode, GlyphError
from .geometry import (
    DEFAULT_SNAP,
    Loop,
    Vec,
    add,
    cross,
    distance,
    mul,
    signed_area,
    sub,
    unit,
)
from .model import canonical_json_bytes


class CapStyle(str, Enum):
    BUTT = "butt"
    ROUND = "round"
    SQUARE = "square"


class JoinStyle(str, Enum):
    MITER = "miter"
    ROUND = "round"
    BEVEL = "bevel"


@dataclass(frozen=True, slots=True)
class StrokeOptions:
    width: float
    cap: CapStyle | str = CapStyle.ROUND
    join: JoinStyle | str = JoinStyle.ROUND
    tolerance: float = 0.1
    miter_limit: float = 4.0
    snap_quantum: float = DEFAULT_SNAP

    def validated(self) -> "StrokeOptions":
        try:
            cap = CapStyle(self.cap)
        except ValueError:
            raise GlyphError(ErrorCode.CAP_JOIN, f"invalid cap: {self.cap}") from None
        try:
            join = JoinStyle(self.join)
        except ValueError:
            raise GlyphError(ErrorCode.CAP_JOIN, f"invalid join: {self.join}") from None
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise GlyphError(ErrorCode.STROKE_WIDTH, "stroke width must be finite and positive", width=self.width)
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise GlyphError(ErrorCode.CAP_JOIN, "stroke tolerance must be finite and positive")
        if not math.isfinite(self.miter_limit) or self.miter_limit <= 0.0:
            raise GlyphError(ErrorCode.CAP_JOIN, "miter limit must be finite and positive")
        return StrokeOptions(
            float(self.width),
            cap,
            join,
            float(self.tolerance),
            float(self.miter_limit),
            float(self.snap_quantum),
        )


@dataclass(frozen=True, slots=True)
class StrokeResult:
    loops: tuple[tuple[Vec, ...], ...]
    direct_loops: tuple[tuple[Vec, ...], ...]
    boundary_hash_value: str
    cap: str
    join: str
    width: float
    tolerance: float
    arc_approximation_error: float
    primitive_count: int
    boolean_result: BooleanResult

    @property
    def boundary_point_count(self) -> int:
        return sum(len(loop) for loop in self.loops)

    def as_payload(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "join": self.join,
            "width": self.width,
            "tolerance": self.tolerance,
            "arc_approximation_error": self.arc_approximation_error,
            "loops": [
                [{"x": float(f"{x:.12f}"), "y": float(f"{y:.12f}")} for x, y in loop]
                for loop in self.loops
            ],
        }

    def boundary_hash(self) -> str:
        return self.boundary_hash_value


def stroke_polyline(
    points: Sequence[Vec],
    options: StrokeOptions,
    *,
    resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
) -> StrokeResult:
    opts = options.validated()
    path = _dedupe(points)
    if len(path) < 2:
        raise GlyphError(ErrorCode.STROKE_WIDTH, "stroke path needs at least two distinct points")

    half = opts.width * 0.5
    primitives: list[Loop] = []
    arc_error = 0.0
    circle_points, circle_error = _circle_polygon((0.0, 0.0), half, opts.tolerance)
    arc_error = max(arc_error, circle_error)

    for start, end in zip(path, path[1:]):
        direction = unit(sub(end, start))
        if direction == (0.0, 0.0):
            continue
        normal = (-direction[1], direction[0])
        rectangle = [
            add(start, mul(normal, -half)),
            add(end, mul(normal, -half)),
            add(end, mul(normal, half)),
            add(start, mul(normal, half)),
        ]
        primitives.append(_ccw(rectangle))

    if opts.cap is CapStyle.ROUND:
        primitives.append(_ccw(_translated_circle(circle_points, path[0])))
        primitives.append(_ccw(_translated_circle(circle_points, path[-1])))
    elif opts.cap is CapStyle.SQUARE:
        d0 = unit(sub(path[0], path[1]))
        d1 = unit(sub(path[-1], path[-2]))
        primitives.append(_ccw(_rectangle_for_segment(add(path[0], mul(d0, half)), path[0], half)))
        primitives.append(_ccw(_rectangle_for_segment(path[-1], add(path[-1], mul(d1, half)), half)))

    for idx in range(1, len(path) - 1):
        previous = path[idx - 1]
        current = path[idx]
        following = path[idx + 1]
        d0 = unit(sub(current, previous))
        d1 = unit(sub(following, current))
        if d0 == (0.0, 0.0) or d1 == (0.0, 0.0):
            continue
        turn = cross(d0, d1)
        if abs(turn) <= 1e-12:
            continue
        if opts.join is JoinStyle.ROUND:
            primitives.append(_ccw(_translated_circle(circle_points, current)))
            continue

        normal0 = (-d0[1], d0[0])
        normal1 = (-d1[1], d1[0])
        side = -1.0 if turn > 0.0 else 1.0
        q0 = add(current, mul(normal0, side * half))
        q1 = add(current, mul(normal1, side * half))

        if opts.join is JoinStyle.MITER:
            miter = _line_intersection(q0, d0, q1, d1)
            if miter is not None and distance(miter, current) <= opts.miter_limit * half + 1e-12:
                primitives.append(_ccw([current, q0, miter, q1]))
                continue
        primitives.append(_ccw([current, q0, q1]))

    if not primitives:
        raise GlyphError(ErrorCode.STROKE_WIDTH, "stroke generated no geometry")

    union = boolean_operation(
        primitives,
        (),
        BooleanOp.UNION,
        resolution_order=resolution_order,
        snap_quantum=opts.snap_quantum,
    )
    if not union.loops:
        raise GlyphError(ErrorCode.STROKE_WIDTH, "stroke union collapsed to empty geometry")
    payload = {
        "cap": opts.cap.value,
        "join": opts.join.value,
        "width": opts.width,
        "tolerance": opts.tolerance,
        "arc_approximation_error": arc_error,
        "loops": [
            [{"x": float(f"{x:.12f}"), "y": float(f"{y:.12f}")} for x, y in loop]
            for loop in union.loops
        ],
    }
    boundary_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return StrokeResult(
        union.loops,
        union.direct_loops,
        boundary_hash,
        opts.cap.value,
        opts.join.value,
        opts.width,
        opts.tolerance,
        arc_error,
        len(primitives),
        union,
    )


def _dedupe(points: Sequence[Vec]) -> list[Vec]:
    result: list[Vec] = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if not result or distance(result[-1], point) > 1e-12:
            result.append(point)
    return result


def _ccw(loop: Sequence[Vec]) -> Loop:
    result = list(loop)
    if signed_area(result) < 0.0:
        result.reverse()
    return result


def _rectangle_for_segment(start: Vec, end: Vec, half: float) -> Loop:
    direction = unit(sub(end, start))
    normal = (-direction[1], direction[0])
    return [
        add(start, mul(normal, -half)),
        add(end, mul(normal, -half)),
        add(end, mul(normal, half)),
        add(start, mul(normal, half)),
    ]


def _translated_circle(template: Sequence[Vec], center: Vec) -> Loop:
    return [add(center, point) for point in template]


def _circle_polygon(center: Vec, radius: float, tolerance: float) -> tuple[Loop, float]:
    if tolerance >= radius:
        count = 12
    else:
        ratio = min(1.0, max(-1.0, 1.0 - tolerance / radius))
        count = int(math.ceil(math.pi / math.acos(ratio)))
    count = max(12, count)
    if count % 2:
        count += 1
    error = radius * (1.0 - math.cos(math.pi / count))
    return (
        [
            (center[0] + radius * math.cos(2.0 * math.pi * idx / count),
             center[1] + radius * math.sin(2.0 * math.pi * idx / count))
            for idx in range(count)
        ],
        error,
    )


def _line_intersection(p: Vec, d: Vec, q: Vec, e: Vec) -> Vec | None:
    denom = cross(d, e)
    if abs(denom) <= 1e-12:
        return None
    t = cross(sub(q, p), e) / denom
    return add(p, mul(d, t))


def stroke_contour(
    contour: Any,
    size_px: int,
    options: StrokeOptions,
    *,
    units_per_em: float = 64.0,
) -> StrokeResult:
    from .flatten import FlattenOptions, flatten_contour

    if contour.closed:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "stroke input must be an open path")
    tolerance = FlattenOptions(size_px=size_px, units_per_em=units_per_em).tolerance()
    flattened = flatten_contour(contour, tolerance, contour_index=0)
    return stroke_polyline(list(flattened.points), options)
