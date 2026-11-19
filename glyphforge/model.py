"""Outline model, validation, and deterministic serialization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

from .errors import ErrorCode, GlyphError


class SegmentType(str, Enum):
    MOVE = "move"
    LINE = "line"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"


CONTROL_COUNT = {
    SegmentType.MOVE: 0,
    SegmentType.LINE: 0,
    SegmentType.QUADRATIC: 1,
    SegmentType.CUBIC: 2,
}


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """One path node.

    ``segment_type`` describes the segment ending at ``point``.  The first node
    is always ``move``.  A closed contour repeats its first point as the final
    node and marks that final node ``closed=True``.
    """

    point: Point
    segment_type: SegmentType
    controls: tuple[Point, ...] = ()
    closed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point.as_dict(),
            "segment_type": self.segment_type.value,
            "controls": [p.as_dict() for p in self.controls],
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class Contour:
    nodes: tuple[OutlineNode, ...]

    @property
    def closed(self) -> bool:
        return bool(self.nodes and self.nodes[-1].closed)

    def validate(self, *, require_closed: bool = False, context: str = "contour") -> None:
        try:
            _validate_contour(self, require_closed=require_closed)
        except GlyphError as exc:
            details = dict(exc.details)
            details["context"] = context
            raise GlyphError(exc.code, exc.message, details) from None

    def as_dict(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "nodes": [n.as_dict() for n in self.nodes],
        }

    def transformed(self, transform: "Affine") -> "Contour":
        return Contour(
            tuple(
                replace(
                    node,
                    point=transform.apply(node.point),
                    controls=tuple(transform.apply(p) for p in node.controls),
                )
                for node in self.nodes
            )
        )


@dataclass(frozen=True, slots=True)
class GlyphCase:
    glyph_id: str
    category: str
    source_a: tuple[Contour, ...]
    source_b: tuple[Contour, ...]
    open_path: Contour
    suggested_sizes: tuple[int, ...] = (12, 16, 24, 32, 48)
    stroke_width: float = 4.0
    stroke_cap: str = "round"
    stroke_join: str = "round"

    def validate(self) -> None:
        if not self.glyph_id:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "glyph_id must not be empty")
        if not self.suggested_sizes or any(int(s) <= 0 for s in self.suggested_sizes):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "suggested_sizes must be positive")
        if not self.source_a or not self.source_b:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "both outline sources are required")
        for source_name, contours in (("source_a", self.source_a), ("source_b", self.source_b)):
            for idx, contour in enumerate(contours):
                contour.validate(require_closed=True, context=f"{source_name}[{idx}]")
        self.open_path.validate(require_closed=False, context="open_path")
        if self.stroke_width <= 0 or not math.isfinite(self.stroke_width):
            raise GlyphError(ErrorCode.STROKE_WIDTH, "stroke width must be finite and positive")
        if self.stroke_cap not in {"butt", "round", "square"}:
            raise GlyphError(ErrorCode.CAP_JOIN, f"invalid cap: {self.stroke_cap}")
        if self.stroke_join not in {"miter", "round", "bevel"}:
            raise GlyphError(ErrorCode.CAP_JOIN, f"invalid join: {self.stroke_join}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "glyph_id": self.glyph_id,
            "category": self.category,
            "source_a": [c.as_dict() for c in self.source_a],
            "source_b": [c.as_dict() for c in self.source_b],
            "open_path": self.open_path.as_dict(),
            "suggested_sizes": list(self.suggested_sizes),
            "stroke_width": self.stroke_width,
            "stroke_cap": self.stroke_cap,
            "stroke_join": self.stroke_join,
        }


@dataclass(frozen=True, slots=True)
class GlyphSuite:
    seed: int
    cases: tuple[GlyphCase, ...]

    def by_id(self, glyph_id: str) -> GlyphCase:
        for case in self.cases:
            if case.glyph_id == glyph_id:
                return case
        raise KeyError(glyph_id)

    def validate(self) -> None:
        seen: set[str] = set()
        for case in self.cases:
            if case.glyph_id in seen:
                raise GlyphError(ErrorCode.OUTLINE_INVALID, "duplicate glyph id", glyph_id=case.glyph_id)
            seen.add(case.glyph_id)
            case.validate()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "glyphforge.suite.v1",
            "seed": self.seed,
            "cases": [case.as_dict() for case in self.cases],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Affine:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, point: Point) -> Point:
        return Point(
            self.a * point.x + self.c * point.y + self.tx,
            self.b * point.x + self.d * point.y + self.ty,
        )


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def move_node(point: Point) -> OutlineNode:
    return OutlineNode(point, SegmentType.MOVE, (), False)


def line_node(point: Point, *, closed: bool = False) -> OutlineNode:
    return OutlineNode(point, SegmentType.LINE, (), closed)


def quadratic_node(point: Point, control: Point, *, closed: bool = False) -> OutlineNode:
    return OutlineNode(point, SegmentType.QUADRATIC, (control,), closed)


def cubic_node(point: Point, control1: Point, control2: Point, *, closed: bool = False) -> OutlineNode:
    return OutlineNode(point, SegmentType.CUBIC, (control1, control2), closed)


def line_contour(points: Sequence[Point] | Sequence[tuple[float, float]], *, closed: bool = True) -> Contour:
    pts = [_as_point(p) for p in points]
    if len(pts) < 2:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "a contour needs at least two points")
    if closed and _same_point(pts[0], pts[-1]):
        pts = pts[:-1]
    nodes: list[OutlineNode] = [move_node(pts[0])]
    for pt in pts[1:]:
        nodes.append(line_node(pt))
    if closed:
        nodes.append(line_node(pts[0], closed=True))
    return Contour(tuple(nodes))


def open_line_path(points: Sequence[Point] | Sequence[tuple[float, float]]) -> Contour:
    pts = [_as_point(p) for p in points]
    if len(pts) < 2:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "an open path needs at least two points")
    return Contour(tuple([move_node(pts[0])] + [line_node(pt) for pt in pts[1:]]))


def _as_point(value: Point | tuple[float, float]) -> Point:
    if isinstance(value, Point):
        return value
    return Point(float(value[0]), float(value[1]))


def _same_point(a: Point, b: Point, eps: float = 1e-9) -> bool:
    return abs(a.x - b.x) <= eps and abs(a.y - b.y) <= eps


def _validate_contour(contour: Contour, *, require_closed: bool) -> None:
    if len(contour.nodes) < 2:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "contour must contain at least two nodes")
    first = contour.nodes[0]
    if first.segment_type is not SegmentType.MOVE or first.controls or first.closed:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "first node must be an unclosed move node")
    for idx, node in enumerate(contour.nodes):
        if not (math.isfinite(node.point.x) and math.isfinite(node.point.y)):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "non-finite node coordinate", node=idx)
        if not isinstance(node.segment_type, SegmentType):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "illegal segment type", node=idx)
        if idx == 0:
            if node.segment_type is not SegmentType.MOVE:
                raise GlyphError(ErrorCode.OUTLINE_INVALID, "first node must be move", node=idx)
        elif node.segment_type is SegmentType.MOVE:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "move is legal only at the first node", node=idx)
        expected = CONTROL_COUNT[node.segment_type]
        if len(node.controls) != expected:
            raise GlyphError(
                ErrorCode.OUTLINE_INVALID,
                "control point count does not match segment type",
                node=idx,
                expected=expected,
                actual=len(node.controls),
            )
        for cidx, control in enumerate(node.controls):
            if not (math.isfinite(control.x) and math.isfinite(control.y)):
                raise GlyphError(
                    ErrorCode.OUTLINE_INVALID,
                    "non-finite control coordinate",
                    node=idx,
                    control=cidx,
                )
        if node.closed and idx != len(contour.nodes) - 1:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "closed marker is legal only on the final node", node=idx)
    is_closed = contour.nodes[-1].closed
    if require_closed and not is_closed:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "closed contour required")
    if is_closed and not _same_point(contour.nodes[0].point, contour.nodes[-1].point):
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "closed contour must repeat its first point")


def iter_segments(contour: Contour) -> Iterable[tuple[int, Point, SegmentType, tuple[Point, ...], Point]]:
    previous = contour.nodes[0].point
    for idx in range(1, len(contour.nodes)):
        node = contour.nodes[idx]
        yield idx - 1, previous, node.segment_type, node.controls, node.point
        previous = node.point
