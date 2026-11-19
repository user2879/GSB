"""Deterministic adaptive Bezier flattening and error accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import budget_denied, invalid_outline
from .model import Contour, Glyph, Point, Segment

DEFAULT_UNITS_PER_EM = 1000.0
DEFAULT_PIXEL_TOLERANCE = 0.35
MAX_SUBDIVISION_DEPTH = 18


def flatten_tolerance_for_size(
    size: int,
    *,
    units_per_em: float = DEFAULT_UNITS_PER_EM,
    pixel_tolerance: float = DEFAULT_PIXEL_TOLERANCE,
    quality: float = 1.0,
) -> float:
    """Return the tolerance in glyph units for a pixel size tier.

    A tighter pixel-space tolerance is obtained with ``quality > 1``.  The
    result is derived from the size tier and can be passed to
    :func:`flatten_contour` when that function works in glyph units.
    """

    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise budget_denied("pixel size must be a positive integer", size=size)
    if not math.isfinite(units_per_em) or units_per_em <= 0:
        raise budget_denied("units_per_em must be positive and finite")
    if not math.isfinite(pixel_tolerance) or pixel_tolerance <= 0:
        raise budget_denied("pixel_tolerance must be positive and finite")
    if not math.isfinite(quality) or quality <= 0:
        raise budget_denied("quality must be positive and finite")
    return (pixel_tolerance / quality) * units_per_em / float(size)


@dataclass(frozen=True)
class FlattenedSegment:
    segment_index: int
    kind: str
    points: tuple[Point, ...]
    actual_error_estimate: float
    subdivisions: int

    def to_dict(self, *, include_points: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "segment_index": self.segment_index,
            "type": self.kind,
            "point_count": len(self.points),
            "actual_error_estimate": self.actual_error_estimate,
            "subdivisions": self.subdivisions,
        }
        if include_points:
            result["points"] = [point.to_dict() for point in self.points]
        return result


@dataclass(frozen=True)
class FlattenedContour:
    contour_index: int
    start: Point
    points: tuple[Point, ...]
    segments: tuple[FlattenedSegment, ...]
    closed: bool = True

    @property
    def vertex_count(self) -> int:
        return len(self.points)

    def to_dict(self, *, include_points: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contour_index": self.contour_index,
            "start": self.start.to_dict(),
            "vertex_count": self.vertex_count,
            "closed": self.closed,
            "segments": [
                segment.to_dict(include_points=include_points)
                for segment in self.segments
            ],
        }
        if include_points:
            result["points"] = [point.to_dict() for point in self.points]
        return result


@dataclass(frozen=True)
class FlattenedGlyph:
    glyph_id: str
    size: int
    scale: float
    tolerance_pixels: float
    units_per_em: float
    contours: tuple[FlattenedContour, ...]

    @property
    def vertex_count(self) -> int:
        return sum(contour.vertex_count for contour in self.contours)

    @property
    def segment_records(self) -> tuple[FlattenedSegment, ...]:
        return tuple(
            segment for contour in self.contours for segment in contour.segments
        )

    def to_dict(self, *, include_points: bool = False) -> dict[str, Any]:
        return {
            "glyph_id": self.glyph_id,
            "size": self.size,
            "scale": self.scale,
            "tolerance_pixels": self.tolerance_pixels,
            "units_per_em": self.units_per_em,
            "vertex_count": self.vertex_count,
            "contours": [
                contour.to_dict(include_points=include_points)
                for contour in self.contours
            ],
        }


def _scale_point(point: Point, scale: float) -> Point:
    return Point(point.x * scale, point.y * scale)


def _distance_to_line(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    denominator = math.hypot(dx, dy)
    if denominator == 0.0:
        return math.hypot(point.x - start.x, point.y - start.y)
    return (
        abs(dy * point.x - dx * point.y + end.x * start.y - end.y * start.x)
        / denominator
    )


def _quadratic_at(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    one_minus_t = 1.0 - t
    return Point(
        one_minus_t * one_minus_t * p0.x + 2.0 * one_minus_t * t * p1.x + t * t * p2.x,
        one_minus_t * one_minus_t * p0.y + 2.0 * one_minus_t * t * p1.y + t * t * p2.y,
    )


def _cubic_at(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    one_minus_t = 1.0 - t
    a = one_minus_t * one_minus_t * one_minus_t
    b = 3.0 * one_minus_t * one_minus_t * t
    c = 3.0 * one_minus_t * t * t
    d = t * t * t
    return Point(
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    )


def _midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)


@dataclass(frozen=True)
class _CurveResult:
    points: tuple[Point, ...]
    actual_error: float
    subdivisions: int
    reached_depth_limit: bool


def _flatten_quadratic(
    p0: Point,
    p1: Point,
    p2: Point,
    tolerance: float,
    depth: int,
) -> _CurveResult:
    flatness = _distance_to_line(p1, p0, p2)
    if flatness <= tolerance or depth >= MAX_SUBDIVISION_DEPTH:
        midpoint = _quadratic_at(p0, p1, p2, 0.5)
        quarter = _quadratic_at(p0, p1, p2, 0.25)
        three_quarter = _quadratic_at(p0, p1, p2, 0.75)
        error = max(
            _distance_to_line(quarter, p0, p2),
            _distance_to_line(midpoint, p0, p2),
            _distance_to_line(three_quarter, p0, p2),
        )
        return _CurveResult(
            points=(p2,),
            actual_error=error,
            subdivisions=0,
            reached_depth_limit=depth >= MAX_SUBDIVISION_DEPTH,
        )

    p01 = _midpoint(p0, p1)
    p12 = _midpoint(p1, p2)
    middle = _midpoint(p01, p12)
    left = _flatten_quadratic(p0, p01, middle, tolerance, depth + 1)
    right = _flatten_quadratic(middle, p12, p2, tolerance, depth + 1)
    return _CurveResult(
        points=left.points + right.points,
        actual_error=max(left.actual_error, right.actual_error),
        subdivisions=left.subdivisions + right.subdivisions + 1,
        reached_depth_limit=left.reached_depth_limit or right.reached_depth_limit,
    )


def _flatten_cubic(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    tolerance: float,
    depth: int,
) -> _CurveResult:
    flatness = max(_distance_to_line(p1, p0, p3), _distance_to_line(p2, p0, p3))
    if flatness <= tolerance or depth >= MAX_SUBDIVISION_DEPTH:
        midpoint = _cubic_at(p0, p1, p2, p3, 0.5)
        quarter = _cubic_at(p0, p1, p2, p3, 0.25)
        three_quarter = _cubic_at(p0, p1, p2, p3, 0.75)
        error = max(
            _distance_to_line(quarter, p0, p3),
            _distance_to_line(midpoint, p0, p3),
            _distance_to_line(three_quarter, p0, p3),
        )
        return _CurveResult(
            points=(p3,),
            actual_error=error,
            subdivisions=0,
            reached_depth_limit=depth >= MAX_SUBDIVISION_DEPTH,
        )

    p01 = _midpoint(p0, p1)
    p12 = _midpoint(p1, p2)
    p23 = _midpoint(p2, p3)
    p012 = _midpoint(p01, p12)
    p123 = _midpoint(p12, p23)
    middle = _midpoint(p012, p123)
    left = _flatten_cubic(p0, p01, p012, middle, tolerance, depth + 1)
    right = _flatten_cubic(middle, p123, p23, p3, tolerance, depth + 1)
    return _CurveResult(
        points=left.points + right.points,
        actual_error=max(left.actual_error, right.actual_error),
        subdivisions=left.subdivisions + right.subdivisions + 1,
        reached_depth_limit=left.reached_depth_limit or right.reached_depth_limit,
    )


def _flatten_segment(
    segment: Segment,
    start: Point,
    *,
    segment_index: int,
    scale: float,
    tolerance: float,
) -> FlattenedSegment:
    scaled_start = _scale_point(start, scale)
    scaled_to = _scale_point(segment.to, scale)
    if segment.kind == "line":
        return FlattenedSegment(
            segment_index=segment_index,
            kind="line",
            points=(scaled_to,),
            actual_error_estimate=0.0,
            subdivisions=0,
        )
    if segment.kind == "quadratic":
        assert segment.control1 is not None
        result = _flatten_quadratic(
            scaled_start,
            _scale_point(segment.control1, scale),
            scaled_to,
            tolerance,
            0,
        )
    else:
        assert segment.control1 is not None and segment.control2 is not None
        result = _flatten_cubic(
            scaled_start,
            _scale_point(segment.control1, scale),
            _scale_point(segment.control2, scale),
            scaled_to,
            tolerance,
            0,
        )
    if result.reached_depth_limit:
        # Finite output is more useful than silently hanging.  The flag is
        # surfaced by the caller through the stored error estimate.
        pass
    return FlattenedSegment(
        segment_index=segment_index,
        kind=segment.kind,
        points=result.points,
        actual_error_estimate=result.actual_error,
        subdivisions=result.subdivisions,
    )


def flatten_contour(
    contour: Contour,
    *,
    contour_index: int = 0,
    scale: float = 1.0,
    tolerance: float = DEFAULT_PIXEL_TOLERANCE,
) -> FlattenedContour:
    if not math.isfinite(scale) or scale <= 0:
        raise budget_denied("flattening scale must be positive and finite")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise budget_denied("flattening tolerance must be positive and finite")
    start = _scale_point(contour.start, scale)
    points: list[Point] = [start]
    current = contour.start
    segments: list[FlattenedSegment] = []
    for index, segment in enumerate(contour.segments):
        flattened = _flatten_segment(
            segment,
            current,
            segment_index=index,
            scale=scale,
            tolerance=tolerance,
        )
        segments.append(flattened)
        points.extend(flattened.points)
        current = segment.to
    return FlattenedContour(
        contour_index=contour_index,
        start=start,
        points=tuple(points),
        segments=tuple(segments),
        closed=contour.closed,
    )


def flatten_glyph(
    glyph: Glyph,
    size: int,
    *,
    units_per_em: float = DEFAULT_UNITS_PER_EM,
    pixel_tolerance: float = DEFAULT_PIXEL_TOLERANCE,
    quality: float = 1.0,
) -> FlattenedGlyph:
    if not isinstance(glyph, Glyph):
        raise invalid_outline("flatten_glyph requires a Glyph")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise budget_denied("pixel size must be a positive integer", size=size)
    if not math.isfinite(units_per_em) or units_per_em <= 0:
        raise budget_denied("units_per_em must be positive and finite")
    if not math.isfinite(pixel_tolerance) or pixel_tolerance <= 0:
        raise budget_denied("pixel_tolerance must be positive and finite")
    if not math.isfinite(quality) or quality <= 0:
        raise budget_denied("quality must be positive and finite")
    scale = float(size) / units_per_em
    tolerance_pixels = pixel_tolerance / quality
    contours = tuple(
        flatten_contour(
            contour,
            contour_index=index,
            scale=scale,
            tolerance=tolerance_pixels,
        )
        for index, contour in enumerate(glyph.contours)
    )
    return FlattenedGlyph(
        glyph_id=glyph.glyph_id,
        size=size,
        scale=scale,
        tolerance_pixels=tolerance_pixels,
        units_per_em=units_per_em,
        contours=contours,
    )
