"""Patch geometry caches and exact local invalidation closures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical import (
    array_to_list,
    content_hash,
    hash_float_array,
    hash_int_array,
    sha256_bytes,
)
from .errors import DomainError, ErrorCode
from .nurbs import EVALUATION_CONVENTION, NURBSPatch


@dataclass(frozen=True)
class InvalidationClosure:
    patch_keys: tuple[str, ...]
    vertices: tuple[int, ...]
    iso_samples: tuple[int, ...]
    triangles: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_keys": list(self.patch_keys),
            "vertices": list(self.vertices),
            "iso_samples": list(self.iso_samples),
            "triangles": list(self.triangles),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvalidationClosure":
        return cls(
            patch_keys=tuple(str(item) for item in value.get("patch_keys", [])),
            vertices=tuple(sorted(int(item) for item in value.get("vertices", []))),
            iso_samples=tuple(sorted(int(item) for item in value.get("iso_samples", []))),
            triangles=tuple(sorted(int(item) for item in value.get("triangles", []))),
        )

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


def _triangle_indices(height: int, width: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for i in range(height - 1):
        for j in range(width - 1):
            a = i * width + j
            b = (i + 1) * width + j
            c = (i + 1) * width + j + 1
            d = i * width + j + 1
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return np.asarray(triangles, dtype=np.int64)


def _build_cp_dependencies(
    grid: Any,
    width: int,
) -> dict[tuple[int, int], tuple[int, ...]]:
    dependencies: dict[tuple[int, int], set[int]] = {}
    for i, row_u in enumerate(grid.support_u):
        for j, support_u in enumerate(row_u):
            support_v = grid.support_v[i][j]
            vertex = i * width + j
            for cp_u in support_u:
                for cp_v in support_v:
                    dependencies.setdefault((int(cp_u), int(cp_v)), set()).add(vertex)
    return {key: tuple(sorted(value)) for key, value in sorted(dependencies.items())}


def _triangles_touching(vertices: Iterable[int], triangles: np.ndarray) -> tuple[int, ...]:
    affected = set(int(item) for item in vertices)
    if not affected:
        return ()
    mask = np.any(np.isin(triangles, np.asarray(sorted(affected), dtype=np.int64)), axis=1)
    return tuple(int(value) for value in np.flatnonzero(mask))


class PatchGeometry:
    """Immutable sampled geometry for one patch and one committed state version."""

    def __init__(
        self,
        *,
        version: int,
        patch_key: str,
        control_hash: str,
        u_values: Sequence[float],
        v_values: Sequence[float],
        points: np.ndarray,
        du: np.ndarray,
        dv: np.ndarray,
        normals: np.ndarray,
        span_u: np.ndarray,
        span_v: np.ndarray,
        triangles: np.ndarray,
        cp_to_vertices: Mapping[tuple[int, int], Sequence[int]],
        evaluation_record: Mapping[str, Any],
        recompute_trace: Mapping[str, Any] | None = None,
    ) -> None:
        self.version = int(version)
        self.patch_key = str(patch_key)
        self.control_hash = str(control_hash)
        self.u_values = tuple(float(item) for item in u_values)
        self.v_values = tuple(float(item) for item in v_values)
        self.points = np.ascontiguousarray(points, dtype="<f8")
        self.du = np.ascontiguousarray(du, dtype="<f8")
        self.dv = np.ascontiguousarray(dv, dtype="<f8")
        self.normals = np.ascontiguousarray(normals, dtype="<f8")
        self.span_u = np.ascontiguousarray(span_u, dtype=np.int64)
        self.span_v = np.ascontiguousarray(span_v, dtype=np.int64)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int64)
        self.cp_to_vertices = {
            (int(item[0]), int(item[1])): tuple(sorted(int(vertex) for vertex in vertices))
            for item, vertices in sorted(cp_to_vertices.items())
        }
        self.evaluation_record = dict(evaluation_record)
        self.recompute_trace = dict(recompute_trace or {})
        self._validate_arrays()
        self._triangle_adjacency: dict[int, tuple[int, ...]] | None = None
        self._recompute_hashes()

    @property
    def height(self) -> int:
        return int(self.points.shape[0])

    @property
    def width(self) -> int:
        return int(self.points.shape[1])

    @property
    def vertex_count(self) -> int:
        return self.height * self.width

    def _validate_arrays(self) -> None:
        expected = (self.height, self.width, 3)
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise DomainError(ErrorCode.GEOM_STALE, "invalid sampled point array", patch_key=self.patch_key)
        if not self.height or not self.width:
            raise DomainError(ErrorCode.GEOM_STALE, "empty sampled geometry", patch_key=self.patch_key)
        if self.du.shape != expected or self.dv.shape != expected or self.normals.shape != expected:
            raise DomainError(ErrorCode.GEOM_STALE, "sampled derivative/normal shape mismatch", patch_key=self.patch_key)
        if self.span_u.shape != (self.height, self.width) or self.span_v.shape != self.span_u.shape:
            raise DomainError(ErrorCode.GEOM_STALE, "span array shape mismatch", patch_key=self.patch_key)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise DomainError(ErrorCode.GEOM_STALE, "triangle index array shape mismatch", patch_key=self.patch_key)
        if len(self.u_values) != self.height or len(self.v_values) != self.width:
            raise DomainError(ErrorCode.GEOM_STALE, "iso parameter arrays do not match sample grid", patch_key=self.patch_key)

    @classmethod
    def build(
        cls,
        patch: NURBSPatch,
        version: int,
        u_values: Sequence[float],
        v_values: Sequence[float],
    ) -> "PatchGeometry":
        grid = patch.evaluate_grid(u_values, v_values)
        width = len(grid.v_values)
        triangles = _triangle_indices(len(grid.u_values), width)
        dependencies = _build_cp_dependencies(grid, width)
        evaluation_record = {
            "degree_u": patch.degree_u,
            "degree_v": patch.degree_v,
            "domain_u": list(patch.domain_u),
            "domain_v": list(patch.domain_v),
            "knot_intervals_u": sorted({(float(patch.knots_u[span]), float(patch.knots_u[span + 1])) for span in range(patch.degree_u, len(patch.knots_u) - patch.degree_u - 1)}),
            "knot_intervals_v": sorted({(float(patch.knots_v[span]), float(patch.knots_v[span + 1])) for span in range(patch.degree_v, len(patch.knots_v) - patch.degree_v - 1)}),
            "sample_count": [len(grid.u_values), len(grid.v_values)],
            "convention": grid.convention,
            "parseval_sample_order": "u_major_then_v",
        }
        return cls(
            version=version,
            patch_key=patch.key,
            control_hash=patch.control_content_hash(),
            u_values=grid.u_values,
            v_values=grid.v_values,
            points=grid.points,
            du=grid.du,
            dv=grid.dv,
            normals=grid.normals,
            span_u=grid.span_u,
            span_v=grid.span_v,
            triangles=triangles,
            cp_to_vertices=dependencies,
            evaluation_record=evaluation_record,
            recompute_trace={"kind": "initial_build", "patch_key": patch.key},
        )

    def _recompute_hashes(self) -> None:
        self.point_hash = hash_float_array("points", self.points)
        self.du_hash = hash_float_array("du", self.du)
        self.dv_hash = hash_float_array("dv", self.dv)
        self.normals_hash = hash_float_array("normals", self.normals)
        self.triangles_hash = hash_int_array("triangles", self.triangles)
        self.iso_hash = sha256_bytes(
            (
                self.point_hash
                + self.du_hash
                + self.dv_hash
                + self.normals_hash
                + content_hash({"u_values": list(self.u_values), "v_values": list(self.v_values)})
            ).encode("ascii")
        )
        self.tessellation_hash = sha256_bytes(
            (self.point_hash + self.normals_hash + self.triangles_hash).encode("ascii")
        )
        self.geometry_hash = content_hash(
            {
                "patch_key": self.patch_key,
                "control_hash": self.control_hash,
                "point_hash": self.point_hash,
                "du_hash": self.du_hash,
                "dv_hash": self.dv_hash,
                "normals_hash": self.normals_hash,
                "triangles_hash": self.triangles_hash,
                "iso_hash": self.iso_hash,
                "evaluation_record": self.evaluation_record,
            }
        )

    def rebind_version(self, new_version: int, *, trace: Mapping[str, Any] | None = None) -> "PatchGeometry":
        """Create a cache entry for a new state without evaluating any geometry."""

        return PatchGeometry(
            version=new_version,
            patch_key=self.patch_key,
            control_hash=self.control_hash,
            u_values=self.u_values,
            v_values=self.v_values,
            points=self.points,
            du=self.du,
            dv=self.dv,
            normals=self.normals,
            span_u=self.span_u,
            span_v=self.span_v,
            triangles=self.triangles,
            cp_to_vertices=self.cp_to_vertices,
            evaluation_record=self.evaluation_record,
            recompute_trace=dict(trace) if trace is not None else {
                "kind": "cache_rebind_only",
                "source_version": self.version,
                "geometry_recomputed": False,
            },
        )

    def _triangle_adjacency_for(self) -> dict[int, tuple[int, ...]]:
        if self._triangle_adjacency is None:
            mapping: dict[int, list[int]] = {}
            for triangle_index, triangle in enumerate(self.triangles.tolist()):
                for vertex in triangle:
                    mapping.setdefault(int(vertex), []).append(triangle_index)
            self._triangle_adjacency = {key: tuple(value) for key, value in mapping.items()}
        return self._triangle_adjacency

    def closure_for_control_point(self, index_u: int, index_v: int) -> InvalidationClosure:
        key = (int(index_u), int(index_v))
        if key not in self.cp_to_vertices:
            raise DomainError(
                ErrorCode.CLOSURE_MISMATCH,
                "control point has no sampled dependency",
                patch_key=self.patch_key,
                control_point=list(key),
            )
        vertices = tuple(int(item) for item in self.cp_to_vertices[key])
        adjacency = self._triangle_adjacency_for()
        triangle_set: set[int] = set()
        for vertex in vertices:
            triangle_set.update(adjacency.get(vertex, ()))
        return InvalidationClosure(
            patch_keys=(self.patch_key,),
            vertices=vertices,
            iso_samples=vertices,
            triangles=tuple(sorted(triangle_set)),
        )

    def _decode_vertex(self, vertex: int) -> tuple[int, int]:
        return divmod(int(vertex), self.width)

    def apply_control_edit(
        self,
        new_patch: NURBSPatch,
        *,
        new_version: int,
        index_u: int,
        index_v: int,
        closure: InvalidationClosure,
        operation_version: int,
    ) -> "PatchGeometry":
        if new_patch.key != self.patch_key:
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "cannot apply a different patch to this geometry cache",
                expected=self.patch_key,
                actual=new_patch.key,
            )
        if new_patch.degree_u != self.evaluation_record["degree_u"] or new_patch.degree_v != self.evaluation_record["degree_v"]:
            raise DomainError(ErrorCode.GEOM_STALE, "patch degrees differ from the cached evaluation", patch_key=self.patch_key)
        if new_patch.control_shape != (max(key[0] for key in self.cp_to_vertices) + 1, max(key[1] for key in self.cp_to_vertices) + 1):
            raise DomainError(ErrorCode.GEOM_STALE, "patch control grid differs from the cached evaluation", patch_key=self.patch_key)

        points = self.points.copy()
        du = self.du.copy()
        dv = self.dv.copy()
        normals = self.normals.copy()
        recomputed_vertices: list[int] = []
        for vertex in closure.vertices:
            i, j = self._decode_vertex(vertex)
            sample = new_patch.evaluate(self.u_values[i], self.v_values[j])
            points[i, j] = sample.point
            du[i, j] = sample.du
            dv[i, j] = sample.dv
            normals[i, j] = sample.normal
            recomputed_vertices.append(vertex)

        actual_triangles = _triangles_touching(recomputed_vertices, self.triangles)
        trace = {
            "kind": "local_control_point_edit",
            "patch_key": self.patch_key,
            "operation_version": int(operation_version),
            "control_point": [int(index_u), int(index_v)],
            "closure": closure.to_dict(),
            "actual_recomputed": {
                "patch_keys": [self.patch_key],
                "vertices": sorted(recomputed_vertices),
                "iso_samples": sorted(recomputed_vertices),
                "triangles": list(actual_triangles),
            },
        }
        updated = PatchGeometry(
            version=new_version,
            patch_key=self.patch_key,
            control_hash=new_patch.control_content_hash(),
            u_values=self.u_values,
            v_values=self.v_values,
            points=points,
            du=du,
            dv=dv,
            normals=normals,
            span_u=self.span_u.copy(),
            span_v=self.span_v.copy(),
            triangles=self.triangles.copy(),
            cp_to_vertices=self.cp_to_vertices,
            evaluation_record=self.evaluation_record,
            recompute_trace=trace,
        )
        updated.verify_closure(self, closure)
        return updated

    def verify_closure(self, previous: "PatchGeometry", closure: InvalidationClosure) -> None:
        if previous.patch_key != self.patch_key:
            raise DomainError(ErrorCode.CLOSURE_MISMATCH, "closure patch key mismatch")
        affected = np.zeros(self.vertex_count, dtype=bool)
        affected[np.asarray(closure.vertices, dtype=np.int64)] = True
        unaffected = ~affected
        for label, old_array, new_array in (
            ("points", previous.points, self.points),
            ("du", previous.du, self.du),
            ("dv", previous.dv, self.dv),
            ("normals", previous.normals, self.normals),
        ):
            old_flat = old_array.reshape(self.vertex_count, 3)[unaffected]
            new_flat = new_array.reshape(self.vertex_count, 3)[unaffected]
            if old_flat.tobytes(order="C") != new_flat.tobytes(order="C"):
                raise DomainError(
                    ErrorCode.CLOSURE_MISMATCH,
                    "geometry outside the invalidation closure changed",
                    patch_key=self.patch_key,
                    array=label,
                )
        recomputed = self.recompute_trace.get("actual_recomputed", {})
        if tuple(recomputed.get("vertices", [])) != tuple(closure.vertices):
            raise DomainError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recomputed vertices differ from the closure",
                patch_key=self.patch_key,
                expected=list(closure.vertices),
                actual=recomputed.get("vertices", []),
            )
        if tuple(recomputed.get("triangles", [])) != tuple(closure.triangles):
            raise DomainError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recomputed triangles differ from the closure",
                patch_key=self.patch_key,
                expected=list(closure.triangles),
                actual=recomputed.get("triangles", []),
            )
        if tuple(recomputed.get("iso_samples", [])) != tuple(closure.iso_samples):
            raise DomainError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recomputed iso samples differ from the closure",
                patch_key=self.patch_key,
                expected=list(closure.iso_samples),
                actual=recomputed.get("iso_samples", []),
            )

    def triangle_points(self) -> np.ndarray:
        return self.points.reshape(self.vertex_count, 3)[self.triangles]

    def triangle_normals(self) -> np.ndarray:
        return self.normals.reshape(self.vertex_count, 3)[self.triangles]

    def iso_segments(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        u_segments: list[tuple[int, int]] = []
        v_segments: list[tuple[int, int]] = []
        for i in range(self.height):
            for j in range(self.width - 1):
                u_segments.append((i * self.width + j, i * self.width + j + 1))
        for j in range(self.width):
            for i in range(self.height - 1):
                v_segments.append((i * self.width + j, (i + 1) * self.width + j))
        return u_segments, v_segments

    def cache_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "patch_key": self.patch_key,
            "control_hash": self.control_hash,
            "u_values": list(self.u_values),
            "v_values": list(self.v_values),
            "points": array_to_list(self.points),
            "du": array_to_list(self.du),
            "dv": array_to_list(self.dv),
            "normals": array_to_list(self.normals),
            "span_u": self.span_u.tolist(),
            "span_v": self.span_v.tolist(),
            "triangles": self.triangles.tolist(),
            "cp_to_vertices": [[key[0], key[1], list(vertices)] for key, vertices in self.cp_to_vertices.items()],
            "evaluation_record": self.evaluation_record,
            "recompute_trace": self.recompute_trace,
            "hashes": {
                "point_hash": self.point_hash,
                "du_hash": self.du_hash,
                "dv_hash": self.dv_hash,
                "normals_hash": self.normals_hash,
                "triangles_hash": self.triangles_hash,
                "iso_hash": self.iso_hash,
                "tessellation_hash": self.tessellation_hash,
                "geometry_hash": self.geometry_hash,
            },
        }

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, Any]) -> "PatchGeometry":
        try:
            dependencies = {
                (int(item[0]), int(item[1])): tuple(int(vertex) for vertex in item[2])
                for item in payload["cp_to_vertices"]
            }
            geometry = cls(
                version=int(payload["version"]),
                patch_key=str(payload["patch_key"]),
                control_hash=str(payload["control_hash"]),
                u_values=payload["u_values"],
                v_values=payload["v_values"],
                points=np.asarray(payload["points"], dtype="<f8"),
                du=np.asarray(payload["du"], dtype="<f8"),
                dv=np.asarray(payload["dv"], dtype="<f8"),
                normals=np.asarray(payload["normals"], dtype="<f8"),
                span_u=np.asarray(payload["span_u"], dtype=np.int64),
                span_v=np.asarray(payload["span_v"], dtype=np.int64),
                triangles=np.asarray(payload["triangles"], dtype=np.int64),
                cp_to_vertices=dependencies,
                evaluation_record=payload["evaluation_record"],
                recompute_trace=payload.get("recompute_trace"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.CACHE_HASH,
                "patch cache payload is malformed",
                error=str(exc),
            ) from exc
        stored_hashes = payload.get("hashes", {})
        actual_hashes = {
            "point_hash": geometry.point_hash,
            "du_hash": geometry.du_hash,
            "dv_hash": geometry.dv_hash,
            "normals_hash": geometry.normals_hash,
            "triangles_hash": geometry.triangles_hash,
            "iso_hash": geometry.iso_hash,
            "tessellation_hash": geometry.tessellation_hash,
            "geometry_hash": geometry.geometry_hash,
        }
        if dict(stored_hashes) != actual_hashes:
            raise DomainError(
                ErrorCode.CACHE_HASH,
                "patch cache internal hash does not match its geometry",
                patch_key=geometry.patch_key,
                version=geometry.version,
            )
        return geometry


def aggregate_geometry_hashes(geometries: Mapping[str, PatchGeometry]) -> dict[str, str]:
    ordered = sorted(geometries.items())
    geometry_hash = content_hash(
        [
            {
                "patch_key": key,
                "geometry_hash": geometry.geometry_hash,
                "normals_hash": geometry.normals_hash,
                "tessellation_hash": geometry.tessellation_hash,
                "iso_hash": geometry.iso_hash,
            }
            for key, geometry in ordered
        ]
    )
    return {
        "geometry_hash": geometry_hash,
        "normals_hash": content_hash([(key, geometry.normals_hash) for key, geometry in ordered]),
        "iso_hash": content_hash([(key, geometry.iso_hash) for key, geometry in ordered]),
        "tessellation_hash": content_hash([(key, geometry.tessellation_hash) for key, geometry in ordered]),
    }
