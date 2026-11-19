"""Deterministic outline flattening and stroke skeleton derivation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import E_CLOSURE_MISMATCH, E_OUTLINE_DEGENERATE, E_STROKE_SAMPLE, NPRError
from .model import Element, Point, Scene
from .util import sha256_json

EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }


@dataclass(frozen=True, slots=True)
class FlatSegment:
    index: int
    kind: str
    points: tuple[Point, ...]
    error_estimate: float
    vertex_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind,
            "points": [[point.x, point.y] for point in self.points],
            "error_estimate": self.error_estimate,
            "vertex_count": self.vertex_count,
        }


@dataclass(frozen=True, slots=True)
class StrokeSkeleton:
    element_id: str
    vertices: tuple[Point, ...]
    widths: tuple[float, ...]
    indices: tuple[int, ...]
    content_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "element_id": self.element_id,
            "vertices": [[point.x, point.y] for point in self.vertices],
            "widths": list(self.widths),
            "indices": list(self.indices),
            "content_hash": self.content_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "element_id": self.element_id,
            "vertices": [[point.x, point.y] for point in self.vertices],
            "widths": list(self.widths),
            "indices": list(self.indices),
        }


@dataclass(frozen=True, slots=True)
class FlatElement:
    element_id: str
    element_hash: str
    layer_id: str
    layer_order: int
    closed: bool
    stroke_width: float
    tolerance: float
    points: tuple[Point, ...]
    indices: tuple[int, ...]
    segments: tuple[FlatSegment, ...]
    bbox: BBox
    geometry_hash: str
    skeleton_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "element_id": self.element_id,
            "element_hash": self.element_hash,
            "layer_id": self.layer_id,
            "layer_order": self.layer_order,
            "closed": self.closed,
            "stroke_width": self.stroke_width,
            "tolerance": self.tolerance,
            "points": [[point.x, point.y] for point in self.points],
            "indices": list(self.indices),
            "segments": [segment.as_dict() for segment in self.segments],
            "bbox": self.bbox.as_dict(),
            "geometry_hash": self.geometry_hash,
            "skeleton_hash": self.skeleton_hash,
        }


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    denominator = math.hypot(dx, dy)
    if denominator <= EPSILON:
        return point.distance_to(start)
    return abs(dy * point.x - dx * point.y + end.x * start.y - end.y * start.x) / denominator


def _all_finite(points: Iterable[Point]) -> bool:
    return all(math.isfinite(point.x) and math.isfinite(point.y) for point in points)


def _polygon_length(points: tuple[Point, ...]) -> float:
    return sum(points[index].distance_to(points[index + 1]) for index in range(len(points) - 1))


def _degenerate_reason(segment: object) -> str | None:
    points = tuple(segment.points)  # type: ignore[attr-defined]
    kind = str(segment.kind)  # type: ignore[attr-defined]
    if not _all_finite(points):
        return "non-finite control point"
    if kind == "line":
        if points[0].distance_to(points[1]) <= EPSILON:
            return "zero-length line"
        return None
    if _polygon_length(points) <= EPSILON:
        return "zero-length control polygon"
    if any(points[index].distance_to(points[index + 1]) <= EPSILON for index in range(len(points) - 1)):
        return "coincident control points"
    start = points[0]
    end = points[-1]
    if start.distance_to(end) <= EPSILON and all(
        _point_line_distance(point, start, end) <= EPSILON for point in points[1:-1]
    ):
        return "collapsed curve"
    return None


def _split_curve(points: tuple[Point, ...]) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """de Casteljau split at t=0.5 with deterministic float operations."""

    levels: list[list[Point]] = [list(points)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            [
                Point((current[index].x + current[index + 1].x) * 0.5, (current[index].y + current[index + 1].y) * 0.5)
                for index in range(len(current) - 1)
            ]
        )
    left = [level[0] for level in levels]
    right = [level[-1] for level in reversed(levels)]
    return tuple(left), tuple(right)


def _curve_flatness(points: tuple[Point, ...]) -> float:
    start = points[0]
    end = points[-1]
    return max(_point_line_distance(point, start, end) for point in points[1:-1])


def _flatten_curve(points: tuple[Point, ...], tolerance: float, min_subdivisions: int, max_subdivisions: int) -> tuple[tuple[Point, ...], float]:
    forced_depth = 0
    while (1 << forced_depth) < max(1, min_subdivisions):
        forced_depth += 1

    accepted: list[tuple[Point, ...]] = []
    errors: list[float] = []

    def visit(control: tuple[Point, ...], depth: int) -> None:
        flatness = _curve_flatness(control)
        if depth < forced_depth and depth < max_subdivisions:
            left, right = _split_curve(control)
            visit(left, depth + 1)
            visit(right, depth + 1)
            return
        if flatness <= tolerance or depth >= max_subdivisions:
            accepted.append(control)
            errors.append(flatness)
            return
        left, right = _split_curve(control)
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(points, 0)
    result: list[Point] = [accepted[0][0]]
    for control in accepted:
        endpoint = control[-1]
        if endpoint.distance_to(result[-1]) > EPSILON:
            result.append(endpoint)
    return tuple(result), max(errors, default=0.0)


def effective_tolerance(sampling_tolerance_px: float, target_viewport: tuple[int, int], canvas_size: tuple[int, int] | None) -> float:
    if sampling_tolerance_px <= 0.0 or not math.isfinite(sampling_tolerance_px):
        raise NPRError(E_STROKE_SAMPLE, "flatness tolerance must be positive and finite", tolerance=sampling_tolerance_px)
    if canvas_size is None:
        return float(sampling_tolerance_px)
    width, height = canvas_size
    viewport_width, viewport_height = target_viewport
    if width <= 0 or height <= 0 or viewport_width <= 0 or viewport_height <= 0:
        raise NPRError(E_STROKE_SAMPLE, "canvas and viewport dimensions must be positive")
    scale = min(viewport_width / width, viewport_height / height)
    return float(sampling_tolerance_px) / max(scale, EPSILON)


def flatten_element(
    element: Element,
    target_viewport: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    tolerance_override: float | None = None,
) -> tuple[FlatElement, StrokeSkeleton]:
    """Flatten one element and derive its skeleton from the same points."""

    if target_viewport is None:
        target_viewport = canvas_size or (640, 480)
    tolerance_px = float(tolerance_override if tolerance_override is not None else element.sampling.tolerance_px)
    tolerance = effective_tolerance(tolerance_px, target_viewport, canvas_size)
    flat_segments: list[FlatSegment] = []
    all_points: list[Point] = []
    skeleton_indices: list[int] = []
    for segment_index, segment in enumerate(element.segments):
        reason = _degenerate_reason(segment)
        if reason is not None:
            raise NPRError(
                E_OUTLINE_DEGENERATE,
                reason,
                element_id=element.id,
                segment_index=segment_index,
                segment_type=segment.kind,
            )
        if segment.kind == "line":
            segment_points = segment.points
            error = 0.0
        else:
            segment_points, error = _flatten_curve(
                segment.points,
                tolerance,
                element.sampling.min_subdivisions,
                element.sampling.max_subdivisions,
            )
        if len(segment_points) < 2:
            raise NPRError(
                E_OUTLINE_DEGENERATE,
                "flattening produced fewer than two points",
                element_id=element.id,
                segment_index=segment_index,
            )
        flat_segments.append(
            FlatSegment(
                index=segment_index,
                kind=segment.kind,
                points=segment_points,
                error_estimate=float(error),
                vertex_count=len(segment_points),
            )
        )
        if all_points and segment_points[0].distance_to(all_points[-1]) <= EPSILON:
            start_index = len(all_points) - 1
            for point in segment_points[1:]:
                all_points.append(point)
        else:
            start_index = len(all_points)
            for point in segment_points:
                all_points.append(point)
        for edge_index in range(len(segment_points) - 1):
            skeleton_indices.append(start_index + edge_index)
            skeleton_indices.append(start_index + edge_index + 1)
    if not all_points:
        raise NPRError(E_OUTLINE_DEGENERATE, "element flattened to an empty outline", element_id=element.id)

    widths = tuple(element.stroke_width * element.sampling.width_scale for _ in all_points)
    indices: list[int] = skeleton_indices
    if element.closed and len(all_points) >= 3:
        first_index = 0
        last_index = len(all_points) - 1
        if all_points[0].distance_to(all_points[-1]) > EPSILON:
            indices.extend((last_index, first_index))

    xs = [point.x for point in all_points]
    ys = [point.y for point in all_points]
    bbox = BBox(min(xs), min(ys), max(xs), max(ys))
    skeleton_payload = {
        "element_id": element.id,
        "vertices": [[point.x, point.y] for point in all_points],
        "widths": list(widths),
        "indices": indices,
    }
    skeleton_hash = sha256_json(skeleton_payload)
    skeleton = StrokeSkeleton(
        element_id=element.id,
        vertices=tuple(all_points),
        widths=widths,
        indices=tuple(indices),
        content_hash=skeleton_hash,
    )
    geometry_payload = {
        "element_id": element.id,
        "element_hash": element.content_hash,
        "tolerance": tolerance,
        "closed": element.closed,
        "points": [[point.x, point.y] for point in all_points],
        "indices": list(indices),
        "segments": [segment.as_dict() for segment in flat_segments],
        "bbox": bbox.as_dict(),
    }
    flat = FlatElement(
        element_id=element.id,
        element_hash=element.content_hash,
        layer_id=element.layer_id,
        layer_order=element.layer_order,
        closed=element.closed,
        stroke_width=element.stroke_width,
        tolerance=tolerance,
        points=tuple(all_points),
        indices=tuple(indices),
        segments=tuple(flat_segments),
        bbox=bbox,
        geometry_hash=sha256_json(geometry_payload),
        skeleton_hash=skeleton_hash,
    )
    return flat, skeleton


@dataclass(frozen=True, slots=True)
class GeometryRecord:
    element_hash: str
    tolerance: float
    flat: FlatElement
    skeleton: StrokeSkeleton


class GeometryStore:
    """Incremental geometry ledger; unchanged records are never rebuilt."""

    def __init__(self) -> None:
        self._records: dict[str, GeometryRecord] = {}
        self.build_counts: dict[str, int] = {}

    @property
    def records(self) -> Mapping[str, GeometryRecord]:
        return self._records

    def get(self, element_id: str) -> GeometryRecord | None:
        return self._records.get(element_id)

    def skeleton_hashes(self) -> dict[str, str]:
        return {element_id: record.flat.skeleton_hash for element_id, record in self._records.items()}

    def geometry_hashes(self) -> dict[str, str]:
        return {element_id: record.flat.geometry_hash for element_id, record in self._records.items()}

    def effective_tolerance(self, element: Element, target_viewport: tuple[int, int], canvas_size: tuple[int, int] | None) -> float:
        return effective_tolerance(element.sampling.tolerance_px, target_viewport, canvas_size)

    def ensure_element(
        self,
        element: Element,
        target_viewport: tuple[int, int],
        canvas_size: tuple[int, int] | None,
    ) -> GeometryRecord:
        tolerance = self.effective_tolerance(element, target_viewport, canvas_size)
        current = self._records.get(element.id)
        if current is not None and current.element_hash == element.content_hash and current.tolerance == tolerance:
            return current
        flat, skeleton = flatten_element(element, target_viewport=target_viewport, canvas_size=canvas_size)
        record = GeometryRecord(element.content_hash, tolerance, flat, skeleton)
        self._records[element.id] = record
        self.build_counts[element.id] = self.build_counts.get(element.id, 0) + 1
        return record

    def sync_scene(
        self,
        scene: Scene,
        target_viewport: tuple[int, int],
        dirty_elements: set[str] | None = None,
    ) -> set[str]:
        """Synchronise cache and return the set of elements actually rebuilt."""

        rebuilt: set[str] = set()
        present = {element.id for element in scene.elements}
        for stale_id in set(self._records) - present:
            del self._records[stale_id]
        for element in scene.elements:
            if dirty_elements is None or element.id in dirty_elements:
                record = self._records.get(element.id)
                if record is not None and record.element_hash == element.content_hash:
                    expected = self.effective_tolerance(element, target_viewport, (scene.canvas_width, scene.canvas_height))
                    if record.tolerance == expected:
                        continue
                self.ensure_element(element, target_viewport, (scene.canvas_width, scene.canvas_height))
                rebuilt.add(element.id)
            else:
                record = self._records.get(element.id)
                if record is None:
                    raise NPRError(
                        E_CLOSURE_MISMATCH,
                        "cached geometry is missing for an element outside the closure",
                        element_id=element.id,
                    )
                if record.element_hash != element.content_hash:
                    raise NPRError(
                        E_CLOSURE_MISMATCH,
                        "element outside the closure has changed",
                        element_id=element.id,
                    )
        return rebuilt

    def export_records(self) -> dict[str, GeometryRecord]:
        return dict(self._records)

    @classmethod
    def from_records(cls, records: Mapping[str, GeometryRecord]) -> "GeometryStore":
        store = cls()
        store._records = dict(records)
        return store


def flatten_scene(scene: Scene, target_viewport: tuple[int, int] | None = None) -> tuple[dict[str, FlatElement], dict[str, StrokeSkeleton]]:
    viewport = target_viewport or (scene.canvas_width, scene.canvas_height)
    flats: dict[str, FlatElement] = {}
    skeletons: dict[str, StrokeSkeleton] = {}
    for element in scene.elements:
        flat, skeleton = flatten_element(element, target_viewport=viewport, canvas_size=(scene.canvas_width, scene.canvas_height))
        flats[element.id] = flat
        skeletons[element.id] = skeleton
    return flats, skeletons


def verify_closure_hashes(
    before_skeleton: Mapping[str, str],
    after_skeleton: Mapping[str, str],
    before_render: Mapping[str, str],
    after_render: Mapping[str, str],
    changed_elements: Iterable[str],
) -> None:
    changed = set(changed_elements)
    for element_id in set(before_skeleton) | set(after_skeleton):
        if element_id in changed:
            continue
        if before_skeleton.get(element_id) != after_skeleton.get(element_id):
            raise NPRError(E_CLOSURE_MISMATCH, "skeleton hash changed outside the closure", element_id=element_id)
    for element_id in set(before_render) | set(after_render):
        if element_id in changed:
            continue
        if before_render.get(element_id) != after_render.get(element_id):
            raise NPRError(E_CLOSURE_MISMATCH, "render hash changed outside the closure", element_id=element_id)
