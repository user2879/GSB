"""Polyline boolean operations through a deterministic planar arrangement.

The implementation follows one fixed pipeline:

1. sort source edges by the declared order;
2. split every edge at all proper, endpoint, and collinear-overlap intersections;
3. deduplicate atomic edges after coordinate snapping;
4. classify both sides of each atomic edge with the non-zero winding rule;
5. orient the surviving boundary with the result interior on the left;
6. reconstruct, merge, orient, rotate, and sort loops.

This handles self-intersections and coincident/collinear edges without a
sequential clip-and-retry procedure, so source order never changes the region.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .errors import ErrorCode, GlyphError
from .geometry import (
    DEFAULT_SNAP,
    Loop,
    Vec,
    add,
    bounds,
    clean_loop,
    distance,
    dot,
    inside_nonzero,
    left_normal,
    loop_perimeter,
    mul,
    normalize_loops,
    segment_intersection_params,
    snap_point,
    sub,
    unit,
)
from .model import canonical_json_bytes


class BooleanOp(str, Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    XOR = "xor"


class ResolutionOrder(str, Enum):
    LEXICOGRAPHIC = "lexicographic"
    SOURCE_THEN_LEX = "source_then_lex"


@dataclass(frozen=True, slots=True)
class RawEdge:
    start: Vec
    end: Vec
    source: str
    source_index: int
    edge_index: int

    def sort_key(self) -> tuple[Any, ...]:
        a, b = sorted((self.start, self.end))
        return (a, b, self.source, self.source_index, self.edge_index)


@dataclass(frozen=True, slots=True)
class AtomicEdge:
    start: Vec
    end: Vec
    sources: tuple[str, ...]

    def key(self) -> tuple[Vec, Vec]:
        if self.start <= self.end:
            return (self.start, self.end)
        return (self.end, self.start)


@dataclass(frozen=True, slots=True)
class BoundaryEdge:
    start: Vec
    end: Vec
    sources: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": _point_payload(self.start),
            "end": _point_payload(self.end),
            "sources": list(self.sources),
        }


@dataclass(frozen=True, slots=True)
class BooleanResult:
    operation: str
    resolution_order: str
    loops: tuple[tuple[Vec, ...], ...]
    direct_loops: tuple[tuple[Vec, ...], ...]
    boundary_edges: tuple[BoundaryEdge, ...]
    snap_quantum: float

    @property
    def area(self) -> float:
        from .geometry import signed_area

        return abs(sum(signed_area(loop) for loop in self.loops))

    @property
    def boundary_point_count(self) -> int:
        return sum(len(loop) for loop in self.loops)

    def as_payload(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "loops": [_loop_payload(loop) for loop in self.loops],
        }

    def boundary_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_payload())).hexdigest()

    def loop_payloads(self) -> list[list[dict[str, float]]]:
        return [[_point_payload(point) for point in loop] for loop in self.loops]


def boolean_operation(
    loops_a: Sequence[Sequence[Vec]],
    loops_b: Sequence[Sequence[Vec]],
    operation: BooleanOp | str,
    *,
    resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
    snap_quantum: float = DEFAULT_SNAP,
) -> BooleanResult:
    op = BooleanOp(operation)
    order = ResolutionOrder(resolution_order)
    clean_a = [clean_loop(loop, quantum=snap_quantum) for loop in loops_a]
    clean_b = [clean_loop(loop, quantum=snap_quantum) for loop in loops_b]
    clean_a = [loop for loop in clean_a if loop]
    clean_b = [loop for loop in clean_b if loop]

    raw_edges = _collect_edges(clean_a, "A", snap_quantum) + _collect_edges(clean_b, "B", snap_quantum)
    if not raw_edges:
        empty: tuple[tuple[Vec, ...], ...] = ()
        return BooleanResult(op.value, order.value, empty, empty, (), snap_quantum)

    raw_edges = _ordered_raw_edges(raw_edges, order)
    breakpoints: list[list[Vec]] = [[edge.start, edge.end] for edge in raw_edges]
    for i in range(len(raw_edges)):
        for j in range(i + 1, len(raw_edges)):
            a = raw_edges[i]
            b = raw_edges[j]
            pairs = segment_intersection_params(a.start, a.end, b.start, b.end)
            for t, u in pairs:
                pa = _point_on_edge(a, t)
                pb = _point_on_edge(b, u)
                pa_snap = snap_point(pa, snap_quantum)
                pb_snap = snap_point(pb, snap_quantum)
                breakpoints[i].append(pa_snap)
                breakpoints[j].append(pb_snap)

    atomic: dict[tuple[Vec, Vec], AtomicEdge] = {}
    for idx, edge in enumerate(raw_edges):
        points = _dedupe_points([snap_point(p, snap_quantum) for p in breakpoints[idx]])
        r = sub(edge.end, edge.start)
        rr = dot(r, r)
        points.sort(key=lambda point: dot(sub(point, edge.start), r) / rr if rr else 0.0)
        for start, end in zip(points, points[1:]):
            if start == end:
                continue
            key = (start, end) if start <= end else (end, start)
            existing = atomic.get(key)
            if existing is None:
                atomic[key] = AtomicEdge(start, end, (edge.source,))
            else:
                sources = tuple(sorted(set(existing.sources) | {edge.source}))
                atomic[key] = AtomicEdge(existing.start, existing.end, sources)

    atomic_edges = list(atomic.values())
    atomic_edges.sort(key=lambda edge: _atomic_sort_key(edge, order))
    all_bounds = bounds([loop for loop in (*clean_a, *clean_b)])
    diagonal = max(1.0, math.hypot(all_bounds[2] - all_bounds[0], all_bounds[3] - all_bounds[1]))
    # The side probe only needs to clear coordinate snapping.  Keeping it much
    # smaller than a pixel also resolves thin strips formed by independently
    # flattened, nearly coincident curves.
    offset_eps = max(snap_quantum * 8.0, diagonal * 1e-12)

    oriented: list[BoundaryEdge] = []
    for edge in atomic_edges:
        direction = unit(sub(edge.end, edge.start))
        if direction == (0.0, 0.0):
            continue
        midpoint = mul(add(edge.start, edge.end), 0.5)
        normal = left_normal(direction)
        left_point = add(midpoint, mul(normal, offset_eps))
        right_point = add(midpoint, mul(normal, -offset_eps))
        left = _result_membership(
            op,
            inside_nonzero(left_point, clean_a),
            inside_nonzero(left_point, clean_b),
        )
        right = _result_membership(
            op,
            inside_nonzero(right_point, clean_a),
            inside_nonzero(right_point, clean_b),
        )
        if left == right:
            continue
        if left:
            oriented.append(BoundaryEdge(edge.start, edge.end, edge.sources))
        else:
            oriented.append(BoundaryEdge(edge.end, edge.start, edge.sources))

    if not oriented:
        empty = ()
        return BooleanResult(op.value, order.value, empty, empty, (), snap_quantum)

    raw_loops = _reconstruct_loops(oriented, snap_quantum)
    loops = normalize_loops(raw_loops, quantum=snap_quantum, merge=True, orient=True)
    direct_loops = normalize_loops(raw_loops, quantum=snap_quantum, merge=False, orient=True)
    return BooleanResult(
        op.value,
        order.value,
        tuple(tuple(loop) for loop in loops),
        tuple(tuple(loop) for loop in direct_loops),
        tuple(sorted(oriented, key=lambda edge: (edge.start, edge.end, edge.sources))),
        snap_quantum,
    )


def _collect_edges(loops: Sequence[Sequence[Vec]], source: str, quantum: float) -> list[RawEdge]:
    result: list[RawEdge] = []
    for source_index, loop in enumerate(loops):
        for edge_index, end in enumerate(loop):
            start = loop[edge_index - 1]
            if distance(start, end) <= quantum:
                continue
            result.append(RawEdge(start, end, source, source_index, edge_index))
    return result


def _ordered_raw_edges(edges: list[RawEdge], order: ResolutionOrder) -> list[RawEdge]:
    if order is ResolutionOrder.SOURCE_THEN_LEX:
        return sorted(edges, key=lambda edge: (edge.source, edge.source_index, edge.edge_index, edge.sort_key()))
    return sorted(edges, key=lambda edge: edge.sort_key())


def _atomic_sort_key(edge: AtomicEdge, order: ResolutionOrder) -> tuple[Any, ...]:
    start, end = edge.key()
    if order is ResolutionOrder.SOURCE_THEN_LEX:
        return (edge.sources, start, end)
    return (start, end, edge.sources)


def _point_on_edge(edge: RawEdge, t: float) -> Vec:
    return add(edge.start, mul(sub(edge.end, edge.start), min(1.0, max(0.0, t))))


def _dedupe_points(points: Iterable[Vec]) -> list[Vec]:
    seen: set[Vec] = set()
    result: list[Vec] = []
    for point in points:
        if point not in seen:
            seen.add(point)
            result.append(point)
    return result


def _result_membership(op: BooleanOp, in_a: bool, in_b: bool) -> bool:
    if op is BooleanOp.UNION:
        return in_a or in_b
    if op is BooleanOp.INTERSECTION:
        return in_a and in_b
    if op is BooleanOp.DIFFERENCE:
        return in_a and not in_b
    if op is BooleanOp.XOR:
        return in_a != in_b
    raise GlyphError(ErrorCode.BOOLEAN_DEGENERATE, f"unknown boolean operation: {op}")


def _reconstruct_loops(edges: Sequence[BoundaryEdge], quantum: float) -> list[Loop]:
    outgoing: dict[Vec, list[int]] = {}
    for idx, edge in enumerate(edges):
        outgoing.setdefault(edge.start, []).append(idx)
    for indices in outgoing.values():
        indices.sort(key=lambda idx: math.atan2(edges[idx].end[1] - edges[idx].start[1], edges[idx].end[0] - edges[idx].start[0]))

    used: set[int] = set()
    loops: list[Loop] = []
    edge_order = sorted(range(len(edges)), key=lambda idx: (edges[idx].start, edges[idx].end))
    for first_index in edge_order:
        if first_index in used:
            continue
        loop: Loop = []
        current_index = first_index
        start_vertex = edges[first_index].start
        closed_loop = False
        for _ in range(len(edges) + 1):
            if current_index in used:
                raise GlyphError(
                    ErrorCode.BOOLEAN_DEGENERATE,
                    "boundary reconstruction revisited an edge",
                    edge=current_index,
                )
            used.add(current_index)
            edge = edges[current_index]
            loop.append(edge.start)
            if edge.end == start_vertex:
                closed_loop = True
                break
            # Pairing is angular and must consider every outgoing edge, not only
            # unused ones: touching loops can share a vertex.
            candidates = list(outgoing.get(edge.end, ()))
            if not candidates:
                if _negligible_open_chain(loop, quantum):
                    break
                raise GlyphError(
                    ErrorCode.BOOLEAN_DEGENERATE,
                    "open boundary chain during reconstruction",
                    vertex=_point_payload(edge.end),
                    edge_start=_point_payload(edge.start),
                    used_edges=len(used),
                    total_edges=len(edges),
                    outgoing_degree=len(outgoing.get(edge.end, ())),
                    incoming_degree=sum(1 for item in edges if item.end == edge.end),
                    nearby_edges=[
                        item.as_dict()
                        for item in edges
                        if item.start == edge.end or item.end == edge.end
                    ],
                    path_vertices=[_point_payload(point) for point in loop],
                )
            reverse_angle = math.atan2(edge.start[1] - edge.end[1], edge.start[0] - edge.end[0])
            angles = {
                idx: math.atan2(edges[idx].end[1] - edges[idx].start[1], edges[idx].end[0] - edges[idx].start[0])
                for idx in candidates
            }
            def cw_delta(idx: int) -> float:
                return (reverse_angle - angles[idx]) % (2.0 * math.pi)

            # A zero clockwise delta is legal for tangent/cusp vertices where
            # two boundary loops touch and continue along collinear edges.
            current_index = min(candidates, key=lambda idx: (cw_delta(idx), idx))
            if current_index in used:
                if _negligible_open_chain(loop, quantum):
                    break
                raise GlyphError(
                    ErrorCode.BOOLEAN_DEGENERATE,
                    "angular boundary pairing selected a consumed edge",
                    edge=current_index,
                    vertex=_point_payload(edge.end),
                )
        else:
            raise GlyphError(ErrorCode.BOOLEAN_DEGENERATE, "boundary reconstruction did not close")

        if not closed_loop:
            # A chain smaller than the coordinate quantum cannot affect an
            # 8-bit raster result.  It is deterministically discarded instead
            # of poisoning an otherwise valid arrangement.
            continue

        if len(loop) < 3:
            raise GlyphError(ErrorCode.BOOLEAN_DEGENERATE, "boundary loop has fewer than three vertices")
        cleaned = clean_loop(loop, quantum=quantum)
        if cleaned:
            loops.append(cleaned)
    if len(used) != len(edges):
        raise GlyphError(
            ErrorCode.BOOLEAN_DEGENERATE,
            "boundary reconstruction left unused edges",
            used=len(used),
            total=len(edges),
        )
    return loops


def _negligible_open_chain(loop: Sequence[Vec], quantum: float) -> bool:
    if len(loop) < 3:
        return True
    threshold = quantum * max(1.0, loop_perimeter(loop)) * 64.0
    from .geometry import signed_area

    return abs(signed_area(loop)) <= threshold


def _point_payload(point: Vec) -> dict[str, float]:
    return {"x": float(f"{point[0]:.12f}"), "y": float(f"{point[1]:.12f}")}


def _loop_payload(loop: Sequence[Vec]) -> list[dict[str, float]]:
    return [_point_payload(point) for point in loop]


def boundary_payload_to_loops(payload: Sequence[Sequence[dict[str, float]]]) -> list[Loop]:
    return [[(float(p["x"]), float(p["y"])) for p in loop] for loop in payload]
