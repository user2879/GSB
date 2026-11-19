"""Deterministic adaptive flattening for line/quadratic/cubic outline segments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .errors import ErrorCode, GlyphError
from .geometry import add, cross, distance, mul, sub, Vec
from .model import Contour, SegmentType, canonical_json_bytes, iter_segments

ZERO_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class FlattenOptions:
    size_px: int
    units_per_em: float = 64.0
    tolerance_scale: float = 1.0
    absolute_tolerance: float | None = None
    max_depth: int = 24

    def tolerance(self) -> float:
        if self.absolute_tolerance is not None:
            tolerance = float(self.absolute_tolerance)
        else:
            if self.size_px <= 0:
                raise GlyphError(ErrorCode.OUTLINE_INVALID, "pixel size must be positive")
            tolerance = 0.35 * self.units_per_em / float(self.size_px) * self.tolerance_scale
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "flattening tolerance must be finite and positive")
        return tolerance


@dataclass(frozen=True, slots=True)
class SegmentFlattenInfo:
    contour_index: int
    segment_index: int
    segment_type: str
    vertex_count: int
    estimated_error: float
    degenerate: bool
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "contour": self.contour_index,
            "segment": self.segment_index,
            "type": self.segment_type,
            "vertices": self.vertex_count,
            "estimated_error": self.estimated_error,
            "degenerate": self.degenerate,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FlattenedContour:
    contour_index: int
    closed: bool
    points: tuple[Vec, ...]
    segments: tuple[SegmentFlattenInfo, ...]

    @property
    def vertex_count(self) -> int:
        return len(self.points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contour": self.contour_index,
            "closed": self.closed,
            "vertex_count": self.vertex_count,
            "segments": [segment.as_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class FlattenedOutline:
    source: str
    tolerance: float
    contours: tuple[FlattenedContour, ...]

    @property
    def loops(self) -> list[list[Vec]]:
        return [list(contour.points) for contour in self.contours if contour.closed]

    @property
    def open_points(self) -> list[Vec]:
        for contour in self.contours:
            if not contour.closed:
                return list(contour.points)
        return []

    @property
    def vertex_count(self) -> int:
        return sum(contour.vertex_count for contour in self.contours)

    @property
    def degenerate_segments(self) -> int:
        return sum(1 for contour in self.contours for segment in contour.segments if segment.degenerate)

    def boundary_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tolerance": self.tolerance,
            "contours": [contour.as_dict() for contour in self.contours],
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.boundary_payload())).hexdigest()


def flatten_contour(
    contour: Contour,
    tolerance: float,
    *,
    contour_index: int = 0,
    max_depth: int = 24,
) -> FlattenedContour:
    contour.validate(require_closed=False)
    all_points: list[Vec] = [contour.nodes[0].point.as_tuple()]
    infos: list[SegmentFlattenInfo] = []

    for segment_index, start, kind, controls, end in iter_segments(contour):
        start_v = start.as_tuple()
        end_v = end.as_tuple()
        control_v = [point.as_tuple() for point in controls]
        degenerate, reason = _degenerate_reason(start_v, control_v, end_v, kind)
        if degenerate:
            infos.append(
                SegmentFlattenInfo(
                    contour_index,
                    segment_index,
                    kind.value,
                    0,
                    0.0,
                    True,
                    reason,
                )
            )
            continue

        if kind is SegmentType.LINE:
            segment_points = [start_v, end_v]
            error = 0.0
        elif kind is SegmentType.QUADRATIC:
            segment_points, error = _flatten_quadratic(start_v, control_v[0], end_v, tolerance, max_depth)
        elif kind is SegmentType.CUBIC:
            segment_points, error = _flatten_cubic(
                start_v, control_v[0], control_v[1], end_v, tolerance, max_depth
            )
        else:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, f"unsupported segment type: {kind}")

        if not segment_points or segment_points[0] != start_v or segment_points[-1] != end_v:
            raise GlyphError(
                ErrorCode.OUTLINE_INVALID,
                "flattening did not preserve segment endpoints",
                contour=contour_index,
                segment=segment_index,
            )
        all_points.extend(segment_points[1:])
        infos.append(
            SegmentFlattenInfo(
                contour_index,
                segment_index,
                kind.value,
                len(segment_points),
                error,
                False,
                None,
            )
        )

    if contour.closed:
        if len(all_points) > 1 and all_points[0] == all_points[-1]:
            all_points.pop()
        deduped: list[Vec] = []
        for point in all_points:
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        all_points = deduped
        if len(all_points) < 3:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "closed contour flattened to fewer than three vertices")
    else:
        deduped = []
        for point in all_points:
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        all_points = deduped
        if len(all_points) < 2:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "open path flattened to fewer than two vertices")

    return FlattenedContour(contour_index, contour.closed, tuple(all_points), tuple(infos))


def flatten_contours(
    contours: Sequence[Contour],
    options: FlattenOptions,
    *,
    source: str,
) -> FlattenedOutline:
    tolerance = options.tolerance()
    flattened = tuple(
        flatten_contour(contour, tolerance, contour_index=idx, max_depth=options.max_depth)
        for idx, contour in enumerate(contours)
    )
    return FlattenedOutline(source, tolerance, flattened)


def _degenerate_reason(
    start: Vec, controls: list[Vec], end: Vec, kind: SegmentType
) -> tuple[bool, str | None]:
    all_points = [start, *controls, end]
    if kind is SegmentType.LINE and distance(start, end) <= ZERO_EPS:
        return True, "zero_length"
    if all(distance(point, start) <= ZERO_EPS for point in all_points[1:]):
        if controls:
            return True, "coincident_control_points"
        return True, "zero_length"
    return False, None


def _quadratic_flatness(p0: Vec, p1: Vec, p2: Vec) -> float:
    chord = sub(p2, p0)
    if distance(p0, p2) <= ZERO_EPS:
        return distance(p1, p0)
    return abs(cross(chord, sub(p1, p0))) / distance(p0, p2)


def _cubic_flatness(p0: Vec, p1: Vec, p2: Vec, p3: Vec) -> float:
    chord = sub(p3, p0)
    chord_len = distance(p0, p3)
    if chord_len <= ZERO_EPS:
        return max(distance(p1, p0), distance(p2, p0))
    d1 = abs(cross(chord, sub(p1, p0))) / chord_len
    d2 = abs(cross(chord, sub(p2, p0))) / chord_len
    return max(d1, d2)


def _subdivide_quadratic(p0: Vec, p1: Vec, p2: Vec) -> tuple[Vec, Vec, Vec, Vec, Vec]:
    p01 = mul(add(p0, p1), 0.5)
    p12 = mul(add(p1, p2), 0.5)
    p012 = mul(add(p01, p12), 0.5)
    return p0, p01, p012, p12, p2


def _subdivide_cubic(p0: Vec, p1: Vec, p2: Vec, p3: Vec) -> tuple[Vec, Vec, Vec, Vec, Vec, Vec, Vec]:
    p01 = mul(add(p0, p1), 0.5)
    p12 = mul(add(p1, p2), 0.5)
    p23 = mul(add(p2, p3), 0.5)
    p012 = mul(add(p01, p12), 0.5)
    p123 = mul(add(p12, p23), 0.5)
    p0123 = mul(add(p012, p123), 0.5)
    return p0, p01, p012, p0123, p123, p23, p3


def _flatten_quadratic(p0: Vec, p1: Vec, p2: Vec, tolerance: float, max_depth: int) -> tuple[list[Vec], float]:
    max_error = 0.0

    def recurse(a: Vec, b: Vec, c: Vec, depth: int) -> list[Vec]:
        nonlocal max_error
        error = _quadratic_flatness(a, b, c)
        if error <= tolerance or depth >= max_depth:
            max_error = max(max_error, error)
            return [a, c]
        left, lc, mid, rc, right = _subdivide_quadratic(a, b, c)
        return recurse(left, lc, mid, depth + 1)[:-1] + recurse(mid, rc, right, depth + 1)

    points = recurse(p0, p1, p2, 0)
    return points, max_error


def _flatten_cubic(
    p0: Vec, p1: Vec, p2: Vec, p3: Vec, tolerance: float, max_depth: int
) -> tuple[list[Vec], float]:
    max_error = 0.0

    def recurse(a: Vec, b: Vec, c: Vec, d: Vec, depth: int) -> list[Vec]:
        nonlocal max_error
        error = _cubic_flatness(a, b, c, d)
        if error <= tolerance or depth >= max_depth:
            max_error = max(max_error, error)
            return [a, d]
        left, b1, c1, mid, c2, b2, right = _subdivide_cubic(a, b, c, d)
        return recurse(left, b1, c1, mid, depth + 1)[:-1] + recurse(mid, c2, b2, right, depth + 1)

    points = recurse(p0, p1, p2, p3, 0)
    return points, max_error


def flattened_point_count_for_scale(contours: Sequence[Contour], size_px: int, scale: float) -> int:
    return flatten_contours(contours, FlattenOptions(size_px=size_px, tolerance_scale=scale), source="probe").vertex_count
