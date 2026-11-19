"""Deterministic closed-shape generation and defect injection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, ShapeSpec
from .errors import BusinessError, ErrorCode
from .hashing import sha256_file, sha256_value, write_json
from .io_ply import read_ply, write_ply
from .mesh import (
    Mesh,
    edge_data,
    edge_diagnostics,
    mesh_components,
    orient_outward,
    signed_volume,
    surface_area,
)


@dataclass(frozen=True)
class GenerationResult:
    baseline: Mesh
    defective: Mesh
    truth: dict[str, Any]


@dataclass(frozen=True)
class BaselineStats:
    vertex_count: int
    face_count: int
    volume: float
    area: float
    component_count: int


def _icosphere(subdivisions: int) -> Mesh:
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        vertex_list = [point.copy() for point in vertices]
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in midpoint_cache:
                point = vertex_list[key[0]] + vertex_list[key[1]]
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(vertex_list)
                vertex_list.append(point)
            return midpoint_cache[key]

        new_faces: list[list[int]] = []
        for a, b, c in faces.tolist():
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend(
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            )
        vertices = np.asarray(vertex_list, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _torus(
    major_radius: float,
    minor_radius: float,
    tube_segments: int,
    ring_segments: int,
) -> Mesh:
    vertices: list[list[float]] = []
    for i in range(tube_segments):
        u = 2.0 * math.pi * i / tube_segments
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        for j in range(ring_segments):
            v = 2.0 * math.pi * j / ring_segments
            radial = major_radius + minor_radius * math.cos(v)
            vertices.append(
                [radial * cos_u, radial * sin_u, minor_radius * math.sin(v)]
            )

    faces: list[list[int]] = []
    for i in range(tube_segments):
        next_i = (i + 1) % tube_segments
        for j in range(ring_segments):
            next_j = (j + 1) % ring_segments
            a = i * ring_segments + j
            b = next_i * ring_segments + j
            c = next_i * ring_segments + next_j
            d = i * ring_segments + next_j
            faces.extend([[a, b, c], [a, c, d]])
    return Mesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64))


def _polygon_area(points: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def _point_in_triangle_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    d1 = cross(a, b, point)
    d2 = cross(b, c, point)
    d3 = cross(c, a, point)
    has_negative = d1 < -1.0e-14 or d2 < -1.0e-14 or d3 < -1.0e-14
    has_positive = d1 > 1.0e-14 or d2 > 1.0e-14 or d3 > 1.0e-14
    return not (has_negative and has_positive)


def _ear_clip(points: np.ndarray) -> list[list[int]]:
    if _polygon_area(points) < 0:
        points = points[::-1].copy()
    indices = list(range(len(points)))
    triangles: list[list[int]] = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > len(points) * len(points) + 16:
            raise BusinessError(ErrorCode.E_INTERNAL, "ear clipping failed for grooved cube")
        found = False
        for offset, current in enumerate(indices):
            previous = indices[offset - 1]
            following = indices[(offset + 1) % len(indices)]
            a, b, c = points[previous], points[current], points[following]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1.0e-14:
                continue
            if any(
                _point_in_triangle_2d(points[other], a, b, c)
                for other in indices
                if other not in (previous, current, following)
            ):
                continue
            triangles.append([previous, current, following])
            indices.pop(offset)
            found = True
            break
        if not found:
            raise BusinessError(ErrorCode.E_INTERNAL, "no ear found for grooved cube outline")
    triangles.append(indices.copy())
    return triangles


def _grooved_cube(half_height: float) -> Mesh:
    outline = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.30, 1.0],
            [0.30, 0.25],
            [-0.30, 0.25],
            [-0.30, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cap_triangles = _ear_clip(outline)
    count = len(outline)
    vertices = np.vstack(
        (
            np.column_stack((outline, np.full(count, -half_height))),
            np.column_stack((outline, np.full(count, half_height))),
        )
    )
    faces: list[list[int]] = []
    for a, b, c in cap_triangles:
        faces.append([a, c, b])
        faces.append([count + a, count + b, count + c])
    for index in range(count):
        following = (index + 1) % count
        bottom_a = index
        bottom_b = following
        top_a = count + index
        top_b = count + following
        faces.extend([[bottom_a, bottom_b, top_b], [bottom_a, top_b, top_a]])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _translated(mesh: Mesh, translate: tuple[float, float, float]) -> Mesh:
    return Mesh(vertices=mesh.vertices + np.asarray(translate), faces=mesh.faces)


def _build_shape(spec: ShapeSpec) -> Mesh:
    if spec.kind == "icosphere":
        mesh = _icosphere(spec.subdivisions)
        mesh = Mesh(vertices=mesh.vertices * spec.radius, faces=mesh.faces)
    elif spec.kind == "torus":
        mesh = _torus(
            spec.major_radius,
            spec.minor_radius,
            spec.tube_segments,
            spec.ring_segments,
        )
    elif spec.kind == "grooved_cube":
        mesh = _grooved_cube(spec.half_height)
    else:
        raise BusinessError(ErrorCode.E_INTERNAL, f"unsupported shape kind: {spec.kind}")
    mesh = Mesh(vertices=mesh.vertices, faces=orient_outward(mesh.vertices, mesh.faces))
    diagnostics = edge_diagnostics(mesh.faces)
    if (
        diagnostics["nonmanifold_edge_count"] != 0
        or diagnostics["boundary_edge_count"] != 0
        or diagnostics["inconsistent_winding_edge_count"] != 0
    ):
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "base shape is not a consistently oriented closed manifold",
            details={"shape": spec.kind, "diagnostics": diagnostics},
        )
    return _translated(mesh, spec.translate)


def build_base_mesh(config: PipelineConfig) -> Mesh:
    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    vertex_offset = 0
    for spec in config.shapes:
        shape = _build_shape(spec)
        vertex_blocks.append(shape.vertices)
        face_blocks.append(shape.faces + vertex_offset)
        vertex_offset += shape.vertex_count
    vertices = np.vstack(vertex_blocks)
    faces = np.vstack(face_blocks)
    faces = orient_outward(vertices, faces)
    diagnostics = edge_diagnostics(faces)
    if (
        diagnostics["nonmanifold_edge_count"] != 0
        or diagnostics["boundary_edge_count"] != 0
        or diagnostics["inconsistent_winding_edge_count"] != 0
    ):
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "combined baseline mesh failed topology checks",
            details=diagnostics,
        )
    return Mesh(vertices=vertices, faces=faces)


def baseline_stats(mesh: Mesh) -> BaselineStats:
    _, components = mesh_components(mesh.faces)
    return BaselineStats(
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
        volume=signed_volume(mesh.vertices, mesh.faces),
        area=surface_area(mesh.vertices, mesh.faces),
        component_count=components,
    )


def _append_vertices(vertices: np.ndarray, points: list[list[float]]) -> tuple[np.ndarray, list[int]]:
    start = len(vertices)
    appended = np.asarray(points, dtype=np.float64)
    ids = list(range(start, start + len(appended)))
    return np.vstack((vertices, appended)), ids


def _append_faces(faces: np.ndarray, new_faces: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    start = len(faces)
    appended = np.asarray(new_faces, dtype=np.int64)
    ids = list(range(start, start + len(appended)))
    return np.vstack((faces, appended)), ids


def _record(
    index: int,
    defect_type: str,
    faces: list[int],
    vertices: list[int],
    criterion: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "defect_id": f"D{index:05d}",
        "injection_index": index,
        "type": defect_type,
        "faces": sorted(int(value) for value in faces),
        "vertices": sorted(int(value) for value in vertices),
        "criterion": criterion,
    }
    if metadata:
        record["metadata"] = metadata
    return record


def inject_defects(base: Mesh, config: PipelineConfig) -> tuple[Mesh, list[dict[str, Any]]]:
    vertices = base.vertices.copy()
    faces = base.faces.copy()
    records: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    base_face_count = len(faces)
    base_edge_data = edge_data(faces)

    for _ in range(config.defects.non_manifold_edge):
        edge_index = rng.randrange(len(base_edge_data.edges))
        a, b = (int(value) for value in base_edge_data.edges[edge_index])
        incident_face = int(base_edge_data.incident_faces[edge_index][0])
        triangle = vertices[faces[incident_face]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        midpoint = (vertices[a] + vertices[b]) * 0.5
        fin_vertex = midpoint + normal * 0.28
        vertices, vertex_ids = _append_vertices(vertices, [fin_vertex.tolist()])
        faces, face_ids = _append_faces(faces, [[a, b, vertex_ids[0]]])
        records.append(
            _record(
                len(records),
                "non_manifold_edge",
                face_ids,
                [a, b, vertex_ids[0]],
                "undirected_edge_incidence_gt_2",
                {"edge": [a, b], "incidence_before": 2, "incidence_after": 3},
            )
        )

    used_vertices = np.unique(faces.reshape(-1))
    if len(used_vertices):
        candidates = used_vertices.astype(int).tolist()
        if config.defects.duplicate_vertex > len(candidates):
            duplicate_sources = [rng.choice(candidates) for _ in range(config.defects.duplicate_vertex)]
        else:
            duplicate_sources = rng.sample(candidates, config.defects.duplicate_vertex)
    else:
        duplicate_sources = []
    for source in duplicate_sources:
        vertices, vertex_ids = _append_vertices(vertices, [vertices[source].tolist()])
        records.append(
            _record(
                len(records),
                "duplicate_vertex",
                [],
                vertex_ids,
                "coincident_within_weld_tolerance",
                {"canonical_vertex": int(source)},
            )
        )

    for zero_index in range(config.defects.zero_area_face):
        origin = np.array([-2.0 + 1.8 * zero_index, -3.2, 2.8], dtype=np.float64)
        points = [origin.tolist(), (origin + [0.65, 0.0, 0.0]).tolist(), (origin + [1.30, 0.0, 0.0]).tolist()]
        vertices, vertex_ids = _append_vertices(vertices, points)
        faces, face_ids = _append_faces(faces, [vertex_ids])
        records.append(
            _record(
                len(records),
                "zero_area_face",
                face_ids,
                vertex_ids,
                "triangle_area_le_geometry_epsilon",
                {"area": 0.0},
            )
        )

    for block_index in range(config.defects.isolated_component):
        center = np.array([4.6 + 1.6 * block_index, 3.0, 3.2], dtype=np.float64)
        points = [
            (center + [0.0, 0.0, 0.32]).tolist(),
            (center + [0.34, 0.0, 0.0]).tolist(),
            (center + [0.0, 0.34, 0.0]).tolist(),
            (center + [0.0, 0.0, -0.32]).tolist(),
        ]
        vertices, vertex_ids = _append_vertices(vertices, points)
        tetra_faces = [
            [vertex_ids[0], vertex_ids[2], vertex_ids[1]],
            [vertex_ids[0], vertex_ids[1], vertex_ids[3]],
            [vertex_ids[1], vertex_ids[2], vertex_ids[3]],
            [vertex_ids[2], vertex_ids[0], vertex_ids[3]],
        ]
        faces, face_ids = _append_faces(faces, tetra_faces)
        records.append(
            _record(
                len(records),
                "isolated_component",
                face_ids,
                vertex_ids,
                "component_face_count_lt_isolated_threshold",
                {"face_count": 4, "threshold": config.isolated_component_face_threshold},
            )
        )

    for crossing_index in range(config.defects.self_intersection):
        origin = np.array([0.0, 3.2 + 2.0 * crossing_index, 3.0], dtype=np.float64)
        points = [
            origin.tolist(),
            (origin + [1.0, 0.0, 0.0]).tolist(),
            (origin + [0.0, 1.0, 0.0]).tolist(),
            (origin + [0.2, 0.2, -0.4]).tolist(),
            (origin + [0.2, 0.2, 0.4]).tolist(),
            (origin + [0.62, 0.36, 0.4]).tolist(),
        ]
        vertices, vertex_ids = _append_vertices(vertices, points)
        crossing_faces = [
            [vertex_ids[0], vertex_ids[1], vertex_ids[2]],
            [vertex_ids[3], vertex_ids[4], vertex_ids[5]],
        ]
        faces, face_ids = _append_faces(faces, crossing_faces)
        records.append(
            _record(
                len(records),
                "self_intersection",
                face_ids,
                vertex_ids,
                "triangle_triangle_intersection_without_shared_vertex",
                {"pair": face_ids},
            )
        )

    defective = Mesh(vertices=vertices, faces=faces)
    if len(records) != config.defects.injection_count:
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "ground-truth record count does not equal configured injection count",
            details={
                "expected": config.defects.injection_count,
                "actual": len(records),
            },
        )
    return defective, records


def _counts_by_type(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        key = str(record["type"])
        result[key] = result.get(key, 0) + 1
    return result


def generate_sample(config: PipelineConfig, output_dir: str | Path) -> GenerationResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = build_base_mesh(config)
    defective, records = inject_defects(baseline, config)
    baseline_path = output / "baseline.ply"
    defective_path = output / "defective.ply"
    truth_path = output / "ground_truth.json"
    write_ply(baseline_path, baseline.vertices, baseline.faces)
    write_ply(defective_path, defective.vertices, defective.faces)

    stats = baseline_stats(baseline)
    expected_removed = sorted(
        {
            int(face)
            for record in records
            if record["type"] != "duplicate_vertex"
            for face in record["faces"]
        }
    )
    truth: dict[str, Any] = {
        "schema_version": 1,
        "generator": {
            "name": "deterministic_open_mesh_defects",
            "version": 1,
            "seed": config.seed,
        },
        "config": config.to_dict(),
        "config_sha256": sha256_value(config.to_dict()),
        "files": {
            "baseline": "baseline.ply",
            "defective": "defective.ply",
            "ground_truth": "ground_truth.json",
        },
        "input_mesh_sha256": sha256_file(defective_path),
        "baseline_ply_sha256": sha256_file(baseline_path),
        "baseline": {
            "vertex_count": stats.vertex_count,
            "face_count": stats.face_count,
            "volume": stats.volume,
            "area": stats.area,
            "component_count": stats.component_count,
        },
        "tolerances": {
            "volume_relative": config.volume_relative_tolerance,
            "area_relative": config.area_relative_tolerance,
            "weld": config.weld_tolerance,
            "boundary": config.boundary_tolerance,
            "self_intersection_epsilon": config.self_intersection_epsilon,
        },
        "injection_count": config.defects.injection_count,
        "defect_record_count": len(records),
        "counts_by_type": _counts_by_type(records),
        "expected_removed_faces": expected_removed,
        "defects": records,
    }
    write_json(truth_path, truth)
    return GenerationResult(baseline=baseline, defective=defective, truth=truth)


def load_sample(output_dir: str | Path) -> tuple[Mesh, dict[str, Any]]:
    output = Path(output_dir)
    try:
        vertices, faces = read_ply(output / "defective.ply")
    except BusinessError as exc:
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "cannot read defective sample", details=exc.to_dict()) from exc
    import json

    try:
        truth = json.loads((output / "ground_truth.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "cannot read ground-truth manifest",
            details={"error": str(exc)},
        ) from exc
    return Mesh(vertices=vertices, faces=faces), truth
