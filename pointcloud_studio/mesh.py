"""Local surface patches, deterministic triangulation and cache serialization."""

from __future__ import annotations

import hashlib
import json
import struct
from types import MappingProxyType
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import ErrorCode, StudioError
from .util import point_quantization_bound, sha256_bytes


@dataclass(frozen=True)
class ReconstructionParams:
    layout_origin: tuple[float, float, float] = (-8.0, -2.0, -8.0)
    patch_size: float = 0.8
    triangulation_cell_size: float = 0.20
    support_radius: float = 0.576
    initial_normal_error_bound_radians: float = 1.0e-4
    position_tolerance: float = 1.0e-5

    def validate(self) -> None:
        if self.patch_size <= 0.0 or self.triangulation_cell_size <= 0.0:
            raise ValueError("patch and triangulation cell sizes must be positive")
        if self.support_radius <= 0.0:
            raise ValueError("support_radius must be positive")
        if self.position_tolerance < 0.0:
            raise ValueError("position_tolerance must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_origin": [float(item) for item in self.layout_origin],
            "patch_size": float(self.patch_size),
            "triangulation_cell_size": float(self.triangulation_cell_size),
            "support_radius": float(self.support_radius),
            "initial_normal_error_bound_radians": float(self.initial_normal_error_bound_radians),
            "position_tolerance": float(self.position_tolerance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReconstructionParams":
        return cls(
            layout_origin=tuple(float(item) for item in value.get("layout_origin", (-8.0, -2.0, -8.0))),
            patch_size=float(value.get("patch_size", 0.8)),
            triangulation_cell_size=float(value.get("triangulation_cell_size", 0.20)),
            support_radius=float(value.get("support_radius", 0.576)),
            initial_normal_error_bound_radians=float(
                value.get("initial_normal_error_bound_radians", 1.0e-4)
            ),
            position_tolerance=float(value.get("position_tolerance", 1.0e-5)),
        )


@dataclass(frozen=True)
class PatchMesh:
    key: str
    axis: int
    sign: int
    cell_i: int
    cell_j: int
    cell_k: int
    vertices: np.ndarray
    triangles: np.ndarray
    point_count: int
    source_points_sha256: str
    content_sha256: str
    chord_height: float
    quantization_bound: float
    normal_fit_residual: float
    normal_angle_bound: float
    normal_position_bound: float
    position_error_upper_bound: float

    @property
    def aabb_min(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def aabb_max(self) -> np.ndarray:
        return self.vertices.max(axis=0)

    def cache_payload(self) -> bytes:
        return encode_patch_cache(self)


@dataclass(frozen=True)
class CombinedMesh:
    patch_keys: tuple[str, ...]
    vertices: np.ndarray
    triangles: np.ndarray
    patch_vertex_offsets: Mapping[str, tuple[int, int]]
    content_sha256: str

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


MESH_CACHE_MAGIC = b"PCSTPCH1"
MESH_CACHE_VERSION = 1
MESH_CACHE_HEADER = struct.Struct("<8sIIIQ")


def _key(axis: int, sign: int, cell_i: int, cell_j: int, cell_k: int) -> str:
    return f"{'XYZ'[axis]}{'+' if sign > 0 else '-'}:{cell_i}:{cell_j}:{cell_k}"


def _parse_key(key: str) -> tuple[int, int, int, int, int]:
    orientation, first, second, third = key.split(":")
    return (
        "XYZ".index(orientation[0]),
        1 if orientation[1] == "+" else -1,
        int(first),
        int(second),
        int(third),
    )


def _axis_vectors(axis: int) -> tuple[np.ndarray, int, int]:
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = 1.0
    u_axis, v_axis = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]
    return normal, u_axis, v_axis


def assign_patches(
    points: np.ndarray,
    normals: np.ndarray,
    params: ReconstructionParams,
) -> tuple[list[str], list[np.ndarray]]:
    params.validate()
    source = np.asarray(points, dtype=np.float64)
    normal_source = np.asarray(normals, dtype=np.float64)
    if source.shape[0] == 0:
        return [], []
    axis = np.argmax(np.abs(normal_source), axis=1).astype(np.int32)
    row = np.arange(source.shape[0])
    sign = np.where(normal_source[row, axis] >= 0.0, 1, -1).astype(np.int32)
    origin = np.asarray(params.layout_origin, dtype=np.float64)
    # Explicit per-row axis lookup keeps the projection convention auditable.
    u_axis = np.empty(source.shape[0], dtype=np.int32)
    v_axis = np.empty(source.shape[0], dtype=np.int32)
    for value, pair in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
        mask = axis == value
        u_axis[mask] = pair[0]
        v_axis[mask] = pair[1]
    cell_i = np.floor((source[row, u_axis] - origin[u_axis]) / params.patch_size).astype(np.int64)
    cell_j = np.floor((source[row, v_axis] - origin[v_axis]) / params.patch_size).astype(np.int64)
    cell_k = np.floor((source[row, axis] - origin[axis]) / params.patch_size).astype(np.int64)

    records = np.column_stack((axis, sign, cell_i, cell_j, cell_k))
    order = np.lexsort((records[:, 4], records[:, 3], records[:, 2], records[:, 1], records[:, 0]))
    sorted_records = records[order]
    if sorted_records.shape[0]:
        changed = np.any(np.diff(sorted_records, axis=0) != 0, axis=1)
        starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
        ends = np.concatenate((starts[1:], [order.size]))
    else:
        starts = np.zeros(0, dtype=np.int64)
        ends = np.zeros(0, dtype=np.int64)
    keys: list[str] = []
    groups: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        record = sorted_records[start]
        keys.append(
            _key(int(record[0]), int(record[1]), int(record[2]), int(record[3]), int(record[4]))
        )
        groups.append(np.sort(order[start:end]).astype(np.int64, copy=False))
    return keys, groups


def boundary_neighbor_keys(key: str, existing_keys: Iterable[str]) -> set[str]:
    axis, sign, cell_i, cell_j, cell_k = _parse_key(key)
    candidates = {
        _key(axis, sign, cell_i - 1, cell_j, cell_k),
        _key(axis, sign, cell_i + 1, cell_j, cell_k),
        _key(axis, sign, cell_i, cell_j - 1, cell_k),
        _key(axis, sign, cell_i, cell_j + 1, cell_k),
    }
    existing = set(existing_keys)
    return candidates & existing


def source_points_hash(points: np.ndarray, normals: np.ndarray) -> str:
    records = np.column_stack(
        (
            np.asarray(points, dtype="<f4"),
            np.asarray(normals, dtype="<f4"),
        )
    ).astype("<f4", copy=False)
    order = np.lexsort(
        (
            records[:, 5],
            records[:, 4],
            records[:, 3],
            records[:, 2],
            records[:, 1],
            records[:, 0],
        )
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(records[order], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def content_hash(key: str, vertices: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"PCSTPATCHMESH1\0")
    digest.update(key.encode("ascii") + b"\0")
    vertex_array = np.ascontiguousarray(vertices, dtype="<f8")
    triangle_array = np.ascontiguousarray(triangles, dtype="<i4")
    digest.update(struct.pack("<QQ", vertex_array.shape[0], triangle_array.shape[0]))
    digest.update(vertex_array.tobytes(order="C"))
    digest.update(triangle_array.tobytes(order="C"))
    return digest.hexdigest()


def _fit_patch_plane(points: np.ndarray) -> tuple[float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = (centered.T @ centered) / max(points.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    residual = float(np.sqrt(max(eigenvalues[0], 0.0)))
    normal = eigenvectors[:, 0]
    length = float(np.linalg.norm(normal))
    if length > 0.0:
        normal = normal / length
    return residual, normal


def _patch_from_group(
    key: str,
    group: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    params: ReconstructionParams,
    neighbor_radius: float,
) -> PatchMesh:
    axis, sign, cell_i, cell_j, cell_k = _parse_key(key)
    _, u_axis, v_axis = _axis_vectors(axis)
    local_points = np.asarray(points[group], dtype=np.float64)
    local_normals = np.asarray(normals[group], dtype=np.float64)
    point_count = int(group.size)
    source_hash = source_points_hash(local_points, local_normals)
    quantization_bound = float(point_quantization_bound(local_points).max())
    residual, fitted_normal = _fit_patch_plane(local_points)
    angle_bound = max(
        params.initial_normal_error_bound_radians,
        float(np.arctan2(residual + quantization_bound, max(0.5 * neighbor_radius, 1.0e-12))),
    )
    angle_bound = min(np.pi / 2.0, angle_bound)
    normal_position_bound = float(2.0 * neighbor_radius * np.sin(angle_bound))

    origin = np.asarray(params.layout_origin, dtype=np.float64)
    step = params.triangulation_cell_size
    bin_u = np.floor((local_points[:, u_axis] - origin[u_axis]) / step).astype(np.int64)
    bin_v = np.floor((local_points[:, v_axis] - origin[v_axis]) / step).astype(np.int64)
    bins = np.column_stack((bin_u, bin_v))
    bin_order = np.lexsort((bins[:, 1], bins[:, 0]))
    sorted_bins = bins[bin_order]
    unique_bins, starts = np.unique(sorted_bins, axis=0, return_index=True)
    ends = np.concatenate((starts[1:], [bin_order.size]))

    vertices = np.empty((unique_bins.shape[0], 3), dtype=np.float64)
    bin_to_vertex: dict[tuple[int, int], int] = {}
    bin_to_points: dict[tuple[int, int], np.ndarray] = {}
    for vertex_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        point_indices = bin_order[start:end]
        bin_key = (int(sorted_bins[start, 0]), int(sorted_bins[start, 1]))
        bin_to_vertex[bin_key] = vertex_index
        bin_to_points[bin_key] = point_indices
        vertices[vertex_index] = local_points[point_indices].mean(axis=0)

    orientation_normal = np.zeros(3, dtype=np.float64)
    orientation_normal[axis] = float(sign)
    triangle_rows: list[tuple[int, int, int]] = []
    for bin_key, vertex_00 in bin_to_vertex.items():
        ci, cj = bin_key
        vertex_10 = bin_to_vertex.get((ci + 1, cj))
        vertex_01 = bin_to_vertex.get((ci, cj + 1))
        vertex_11 = bin_to_vertex.get((ci + 1, cj + 1))
        if vertex_10 is not None and vertex_01 is not None and vertex_11 is not None:
            if (ci + cj) & 1:
                triangle_rows.extend(((vertex_00, vertex_10, vertex_01), (vertex_10, vertex_11, vertex_01)))
            else:
                triangle_rows.extend(((vertex_00, vertex_10, vertex_11), (vertex_00, vertex_11, vertex_01)))
        elif vertex_10 is not None and vertex_11 is not None:
            triangle_rows.append((vertex_00, vertex_10, vertex_11))
        elif vertex_01 is not None and vertex_11 is not None:
            triangle_rows.append((vertex_00, vertex_11, vertex_01))

    triangles = np.asarray(triangle_rows, dtype=np.int32).reshape(-1, 3)
    if triangles.shape[0]:
        for index in range(triangles.shape[0]):
            a, b, c = triangles[index]
            face = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
            if float(np.dot(face, orientation_normal)) < 0.0:
                triangles[index, 1], triangles[index, 2] = triangles[index, 2], triangles[index, 1]
        triangle_order = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
        triangles = triangles[triangle_order]

    bin_triangles: dict[tuple[int, int], set[int]] = {item: set() for item in bin_to_vertex}
    vertex_bin = {vertex: bin_key for bin_key, vertex in bin_to_vertex.items()}
    for triangle_index, triangle in enumerate(triangles):
        for vertex in triangle:
            bin_triangles[vertex_bin[int(vertex)]].add(triangle_index)

    chord_height = 0.0
    for bin_key, point_indices in bin_to_points.items():
        incident = sorted(bin_triangles[bin_key])
        if incident:
            incident_vertices = np.unique(triangles[np.asarray(incident, dtype=np.int64)].reshape(-1))
            local_plane = vertices[incident_vertices]
            centroid = local_plane.mean(axis=0)
            centered = local_plane - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            plane_normal = vh[-1]
            distance = np.abs((local_points[point_indices] - centroid) @ plane_normal)
            chord_height = max(chord_height, float(distance.max()))
        else:
            distance = np.linalg.norm(local_points[point_indices] - vertices[bin_to_vertex[bin_key]], axis=1)
            chord_height = max(chord_height, float(distance.max()))

    if not triangles.shape[0]:
        vertices = np.zeros((0, 3), dtype=np.float64)
        triangles = np.zeros((0, 3), dtype=np.int32)
        chord_height = 0.0

    vertices.setflags(write=False)
    triangles.setflags(write=False)
    mesh_hash = content_hash(key, vertices, triangles)
    total_bound = quantization_bound + normal_position_bound + chord_height
    return PatchMesh(
        key=key,
        axis=axis,
        sign=sign,
        cell_i=cell_i,
        cell_j=cell_j,
        cell_k=cell_k,
        vertices=vertices,
        triangles=triangles,
        point_count=point_count,
        source_points_sha256=source_hash,
        content_sha256=mesh_hash,
        chord_height=chord_height,
        quantization_bound=quantization_bound,
        normal_fit_residual=residual,
        normal_angle_bound=angle_bound,
        normal_position_bound=normal_position_bound,
        position_error_upper_bound=total_bound,
    )


def build_patches_for_keys(
    keys: Iterable[str],
    groups: Mapping[str, np.ndarray],
    points: np.ndarray,
    normals: np.ndarray,
    params: ReconstructionParams,
    neighbor_radius: float,
) -> dict[str, PatchMesh]:
    result: dict[str, PatchMesh] = {}
    for key in sorted(set(keys), key=lambda item: item.encode("ascii")):
        group = groups.get(key)
        if group is None or group.size < 3:
            continue
        patch = _patch_from_group(key, group, points, normals, params, neighbor_radius)
        if patch.triangles.shape[0]:
            result[key] = patch
    return result


def build_all_patches(
    points: np.ndarray,
    normals: np.ndarray,
    params: ReconstructionParams,
    neighbor_radius: float,
) -> tuple[dict[str, PatchMesh], list[str], list[np.ndarray]]:
    keys, groups = assign_patches(points, normals, params)
    group_map = dict(zip(keys, groups, strict=True))
    patches = build_patches_for_keys(keys, group_map, points, normals, params, neighbor_radius)
    return patches, keys, groups


def combine_mesh(patches: Mapping[str, PatchMesh]) -> CombinedMesh:
    keys = tuple(sorted((key for key, patch in patches.items() if patch.triangles.shape[0]), key=lambda item: item.encode("ascii")))
    if not keys:
        vertices = np.zeros((0, 3), dtype=np.float64)
        triangles = np.zeros((0, 3), dtype=np.int32)
        offsets: dict[str, tuple[int, int]] = {}
    else:
        vertex_chunks: list[np.ndarray] = []
        triangle_chunks: list[np.ndarray] = []
        offsets = {}
        vertex_offset = 0
        for key in keys:
            patch = patches[key]
            offsets[key] = (vertex_offset, vertex_offset + patch.vertices.shape[0])
            vertex_chunks.append(patch.vertices)
            triangle_chunks.append(patch.triangles.astype(np.int64, copy=False) + vertex_offset)
            vertex_offset += patch.vertices.shape[0]
        vertices = np.concatenate(vertex_chunks, axis=0)
        triangles = np.concatenate(triangle_chunks, axis=0).astype(np.int32, copy=False)
    vertices.setflags(write=False)
    triangles.setflags(write=False)
    digest = hashlib.sha256()
    digest.update(b"PCSTMESH1\0")
    for key in keys:
        digest.update(key.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(patches[key].content_sha256))
    return CombinedMesh(
        patch_keys=keys,
        vertices=vertices,
        triangles=triangles,
        patch_vertex_offsets=MappingProxyType(offsets),
        content_sha256=digest.hexdigest(),
    )


def encode_patch_cache(patch: PatchMesh) -> bytes:
    metadata = {
        "schema": "pointcloud-studio.patch-cache/1",
        "key": patch.key,
        "axis": patch.axis,
        "sign": patch.sign,
        "cell_i": patch.cell_i,
        "cell_j": patch.cell_j,
        "cell_k": patch.cell_k,
        "point_count": patch.point_count,
        "source_points_sha256": patch.source_points_sha256,
        "content_sha256": patch.content_sha256,
        "chord_height": patch.chord_height,
        "quantization_bound": patch.quantization_bound,
        "normal_fit_residual": patch.normal_fit_residual,
        "normal_angle_bound": patch.normal_angle_bound,
        "normal_position_bound": patch.normal_position_bound,
        "position_error_upper_bound": patch.position_error_upper_bound,
    }
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    vertices = np.ascontiguousarray(patch.vertices, dtype="<f8")
    triangles = np.ascontiguousarray(patch.triangles, dtype="<i4")
    return (
        MESH_CACHE_HEADER.pack(
            MESH_CACHE_MAGIC,
            MESH_CACHE_VERSION,
            len(metadata_bytes),
            vertices.shape[0],
            triangles.shape[0],
        )
        + metadata_bytes
        + vertices.tobytes(order="C")
        + triangles.tobytes(order="C")
    )


def decode_patch_cache(payload: bytes, *, expected_content_hash: str | None = None, path: str | None = None) -> PatchMesh:
    if len(payload) < MESH_CACHE_HEADER.size:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "truncated patch-cache header",
            context={"path": path, "size": len(payload)},
        )
    magic, version, metadata_length, vertex_count, triangle_count = MESH_CACHE_HEADER.unpack_from(payload, 0)
    if magic != MESH_CACHE_MAGIC or version != MESH_CACHE_VERSION:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "invalid patch-cache header",
            context={"path": path, "magic": magic.hex(), "version": version},
        )
    expected_size = (
        MESH_CACHE_HEADER.size
        + metadata_length
        + vertex_count * 3 * 8
        + triangle_count * 3 * 4
    )
    if len(payload) != expected_size:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch-cache payload size mismatch",
            context={"path": path, "size": len(payload), "expected": expected_size},
        )
    metadata_start = MESH_CACHE_HEADER.size
    vertex_start = metadata_start + metadata_length
    triangle_start = vertex_start + vertex_count * 3 * 8
    try:
        metadata = json.loads(payload[metadata_start:vertex_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "invalid patch-cache metadata",
            context={"path": path, "reason": str(exc)},
        ) from exc
    required_metadata = {
        "key",
        "axis",
        "sign",
        "cell_i",
        "cell_j",
        "cell_k",
        "point_count",
        "source_points_sha256",
        "content_sha256",
        "chord_height",
        "quantization_bound",
        "normal_fit_residual",
        "normal_angle_bound",
        "normal_position_bound",
        "position_error_upper_bound",
    }
    if not isinstance(metadata, dict):
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch-cache metadata is not an object",
            context={"path": path},
        )
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch-cache metadata is incomplete",
            context={"path": path, "missing_fields": missing_metadata},
        )
    vertices = np.frombuffer(payload, dtype="<f8", count=vertex_count * 3, offset=vertex_start).reshape(-1, 3).astype(
        np.float64, copy=True
    )
    triangles = np.frombuffer(payload, dtype="<i4", count=triangle_count * 3, offset=triangle_start).reshape(-1, 3).astype(
        np.int32, copy=True
    )
    actual_hash = content_hash(str(metadata["key"]), vertices, triangles)
    claimed_hash = str(metadata["content_sha256"])
    expected = expected_content_hash or claimed_hash
    if actual_hash != claimed_hash or actual_hash != expected:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch-cache content hash mismatch",
            context={
                "path": path,
                "key": metadata.get("key"),
                "expected": expected,
                "claimed": claimed_hash,
                "actual": actual_hash,
            },
        )
    vertices.setflags(write=False)
    triangles.setflags(write=False)
    try:
        return PatchMesh(
            key=str(metadata["key"]),
            axis=int(metadata["axis"]),
            sign=int(metadata["sign"]),
            cell_i=int(metadata["cell_i"]),
            cell_j=int(metadata["cell_j"]),
            cell_k=int(metadata["cell_k"]),
            vertices=vertices,
            triangles=triangles,
            point_count=int(metadata["point_count"]),
            source_points_sha256=str(metadata["source_points_sha256"]),
            content_sha256=claimed_hash,
            chord_height=float(metadata["chord_height"]),
            quantization_bound=float(metadata["quantization_bound"]),
            normal_fit_residual=float(metadata["normal_fit_residual"]),
            normal_angle_bound=float(metadata["normal_angle_bound"]),
            normal_position_bound=float(metadata["normal_position_bound"]),
            position_error_upper_bound=float(metadata["position_error_upper_bound"]),
        )
    except (TypeError, ValueError) as exc:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch-cache metadata value is malformed",
            context={"path": path, "reason": str(exc)},
        ) from exc


def load_patch_cache(path: str | Path, expected_content_hash: str | None = None) -> PatchMesh:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise StudioError(
            ErrorCode.E_CACHE_HASH,
            "patch cache cannot be read",
            context={"path": str(source), "reason": str(exc)},
        ) from exc
    return decode_patch_cache(payload, expected_content_hash=expected_content_hash, path=str(source))


def patch_cache_name(key: str) -> str:
    safe = key.replace(":", "_").replace("+", "p").replace("-", "m")
    return f"patch_{safe}.pch"
