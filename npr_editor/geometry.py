"""Shared deterministic contour flattening and stroke-skeleton generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .errors import NprError
from .model import Project, Shape, StrokeSample, canonical_hash


@dataclass(frozen=True)
class FlattenedContour:
    closed: bool
    vertices: np.ndarray
    vertex_offset: int

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


@dataclass(frozen=True)
class FlattenedShape:
    shape_id: str
    layer: str
    stroke_width: float
    signature: str
    contours: tuple[FlattenedContour, ...]
    vertices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def geometry_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.vertices.astype("<f4", copy=False).tobytes(order="C"))
        for contour in self.contours:
            digest.update(b"\x01" if contour.closed else b"\x00")
            digest.update(np.asarray([contour.vertex_count], dtype="<u4").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class SkeletonContourRange:
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    closed: bool


@dataclass(frozen=True)
class StrokeSkeleton:
    shape_id: str
    layer: str
    stroke_width: float
    vertices: np.ndarray
    indices: np.ndarray
    contour_ranges: tuple[SkeletonContourRange, ...]
    flattened_hash: str
    signature: str

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def index_count(self) -> int:
        return int(self.indices.shape[0])

    def geometry_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.flattened_hash.encode("ascii"))
        digest.update(self.indices.astype("<u4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def paths(self) -> list[np.ndarray]:
        """Return one point array per contour, including closure vertices."""

        result: list[np.ndarray] = []
        for contour in self.contour_ranges:
            indices = self.indices[
                contour.index_offset : contour.index_offset + contour.index_count
            ]
            result.append(self.vertices[np.asarray(indices, dtype=np.int64)])
        return result


def _sample_line(points: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[:, None]
    sampled = (1.0 - t) * points[0] + t * points[1]
    return sampled.astype(np.float32)


def _sample_quadratic(points: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[:, None]
    one_minus = 1.0 - t
    sampled = (
        (one_minus * one_minus) * points[0]
        + (2.0 * one_minus * t) * points[1]
        + (t * t) * points[2]
    )
    return sampled.astype(np.float32)


def _sample_cubic(points: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[:, None]
    one_minus = 1.0 - t
    sampled = (
        (one_minus**3) * points[0]
        + (3.0 * one_minus * one_minus * t) * points[1]
        + (3.0 * one_minus * t * t) * points[2]
        + (t**3) * points[3]
    )
    return sampled.astype(np.float32)


def sample_segment(segment_type: str, points: np.ndarray, sample: StrokeSample) -> np.ndarray:
    if segment_type == "line":
        return _sample_line(points, sample.line_steps)
    if segment_type == "quadratic":
        return _sample_quadratic(points, sample.curve_steps)
    if segment_type == "cubic":
        return _sample_cubic(points, sample.curve_steps)
    raise NprError(
        "E_SCHEMA_VERSION", "unknown segment type during flattening", type=segment_type
    )


def flatten_shape(shape: Shape, sample: StrokeSample) -> FlattenedShape:
    signature = canonical_hash(
        {"geometry": shape.geometry_dict(), "sample": sample.to_dict()}
    )
    flattened_contours: list[FlattenedContour] = []
    global_chunks: list[np.ndarray] = []
    offset = 0

    for contour in shape.contours:
        chunks: list[np.ndarray] = []
        previous: np.ndarray | None = None
        for segment in contour.segments:
            control = np.asarray(segment.points, dtype=np.float32)
            sampled = sample_segment(segment.type, control, sample)
            if previous is not None and sampled.shape[0] > 0:
                if np.array_equal(sampled[0], previous):
                    sampled = sampled[1:]
            if sampled.shape[0]:
                chunks.append(sampled)
                previous = sampled[-1]
        if not chunks:
            raise NprError(
                "E_STROKE_SAMPLE",
                "stroke sampling produced an empty contour",
                shape=shape.id,
            )
        vertices = np.concatenate(chunks, axis=0)
        if contour.closed and vertices.shape[0] > 1 and np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        if vertices.shape[0] < 2:
            raise NprError(
                "E_STROKE_SAMPLE",
                "stroke sampling produced fewer than two vertices",
                shape=shape.id,
            )
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        vertices.setflags(write=False)
        flattened_contours.append(
            FlattenedContour(contour.closed, vertices, offset)
        )
        global_chunks.append(vertices)
        offset += int(vertices.shape[0])

    all_vertices = np.ascontiguousarray(np.concatenate(global_chunks, axis=0), dtype=np.float32)
    all_vertices.setflags(write=False)
    return FlattenedShape(
        shape_id=shape.id,
        layer=shape.layer,
        stroke_width=float(shape.stroke_width),
        signature=signature,
        contours=tuple(flattened_contours),
        vertices=all_vertices,
    )


def build_skeleton(flattened: FlattenedShape) -> StrokeSkeleton:
    index_chunks: list[np.ndarray] = []
    ranges: list[SkeletonContourRange] = []
    index_offset = 0
    for contour in flattened.contours:
        start = contour.vertex_offset
        stop = start + contour.vertex_count
        indices = list(range(start, stop))
        if contour.closed:
            indices.append(start)
        index_array = np.asarray(indices, dtype=np.uint32)
        index_chunks.append(index_array)
        ranges.append(
            SkeletonContourRange(
                vertex_offset=start,
                vertex_count=contour.vertex_count,
                index_offset=index_offset,
                index_count=int(index_array.shape[0]),
                closed=contour.closed,
            )
        )
        index_offset += int(index_array.shape[0])
    all_indices = np.ascontiguousarray(np.concatenate(index_chunks), dtype=np.uint32)
    all_indices.setflags(write=False)
    skeleton = StrokeSkeleton(
        shape_id=flattened.shape_id,
        layer=flattened.layer,
        stroke_width=flattened.stroke_width,
        vertices=flattened.vertices,
        indices=all_indices,
        contour_ranges=tuple(ranges),
        flattened_hash=flattened.geometry_hash(),
        signature=flattened.signature,
    )
    return skeleton


class GeometryEngine:
    """Signature-keyed geometry cache that shares one skeleton across modes."""

    def __init__(self, project: Project):
        self.project = project
        self._flattened: dict[str, FlattenedShape] = {}
        self._skeletons: dict[str, StrokeSkeleton] = {}
        self.flatten_builds = 0
        self.skeleton_builds = 0
        self.cache_hits = 0

    def reset_stats(self) -> None:
        self.flatten_builds = 0
        self.skeleton_builds = 0
        self.cache_hits = 0

    def set_project(self, project: Project) -> None:
        self.project = project

    def invalidate(self, shape_id: str) -> None:
        self._flattened.pop(shape_id, None)
        self._skeletons.pop(shape_id, None)

    def ensure(self, shape_id: str) -> StrokeSkeleton:
        try:
            shape = self.project.shape_by_id(shape_id)
        except KeyError as exc:
            raise NprError(
                "E_SCHEMA_VERSION", "shape id does not exist", shape_id=shape_id
            ) from exc
        expected_signature = canonical_hash(
            {
                "geometry": shape.geometry_dict(),
                "sample": self.project.stroke_sample.to_dict(),
            }
        )
        flattened = self._flattened.get(shape_id)
        skeleton = self._skeletons.get(shape_id)
        if flattened is not None and skeleton is not None and flattened.signature == expected_signature:
            self.cache_hits += 1
            return skeleton
        flattened = flatten_shape(shape, self.project.stroke_sample)
        self.flatten_builds += 1
        self._flattened[shape_id] = flattened
        skeleton = build_skeleton(flattened)
        self.skeleton_builds += 1
        self._skeletons[shape_id] = skeleton
        return skeleton

    def flattened(self, shape_id: str) -> FlattenedShape:
        self.ensure(shape_id)
        return self._flattened[shape_id]

    def prepare_all(self) -> list[StrokeSkeleton]:
        return [self.ensure(shape.id) for shape in self.project.ordered_shapes()]

    def skeletons(self) -> dict[str, StrokeSkeleton]:
        return dict(self._skeletons)

    def geometry_hashes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for shape in self.project.ordered_shapes():
            result[shape.id] = self.ensure(shape.id).geometry_hash()
        return result

    def shared_paths(self, shape_id: str) -> list[np.ndarray]:
        return self.ensure(shape_id).paths()


def iter_flat_vertices(project: Project, sample: StrokeSample) -> Iterable[tuple[str, np.ndarray]]:
    for shape in project.ordered_shapes():
        flattened = flatten_shape(shape, sample)
        yield shape.id, flattened.vertices
