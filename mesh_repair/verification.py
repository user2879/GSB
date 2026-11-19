"""Independent file-level verifier.

The verifier intentionally starts from the final PLY bytes and the ground-truth
JSON only.  It does not receive block results, repair plans or in-memory mesh
objects from the pipeline.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode
from .hashing import sha256_file
from .io_ply import read_ply
from .mesh import triangle_areas, triangles_intersect


def _python_edges(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, face in enumerate(faces.tolist()):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (a, b) if a < b else (b, a)
            direction = 1 if a < b else -1 if a > b else 0
            result.setdefault(key, []).append((face_id, direction))
    return result


def _python_components(faces: np.ndarray) -> tuple[list[int], int]:
    count = len(faces)
    if count == 0:
        return [], 0
    parent = list(range(count))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for incidents in _python_edges(faces).values():
        if len(incidents) > 1:
            first = incidents[0][0]
            for face_id, _direction in incidents[1:]:
                union(first, face_id)
    roots = [find(index) for index in range(count)]
    unique = {root: index for index, root in enumerate(sorted(set(roots)))}
    labels = [unique[root] for root in roots]
    return labels, len(unique)


def _python_self_intersections(
    vertices: np.ndarray, faces: np.ndarray, eps: float
) -> list[tuple[int, int]]:
    if len(faces) < 2:
        return []
    triangles = vertices[faces]
    minima = np.min(triangles, axis=1) - eps
    maxima = np.max(triangles, axis=1) + eps
    areas = triangle_areas(vertices, faces)
    pairs: list[tuple[int, int]] = []
    for first in range(len(faces) - 1):
        if areas[first] <= eps:
            continue
        overlap = np.all(
            (minima[first] <= maxima[first + 1 :])
            & (maxima[first] >= minima[first + 1 :]),
            axis=1,
        )
        first_vertices = set(int(value) for value in faces[first])
        for offset in np.nonzero(overlap)[0]:
            second = first + 1 + int(offset)
            if areas[second] <= eps:
                continue
            if first_vertices.intersection(int(value) for value in faces[second]):
                continue
            if triangles_intersect(triangles[first], triangles[second], eps):
                pairs.append((first, second))
    return pairs


def _check(
    name: str,
    passed: bool,
    *,
    criterion: str,
    observed: Any,
    expected: Any,
    location: dict[str, Any] | None = None,
    code: ErrorCode,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "criterion": criterion,
        "observed": observed,
        "expected": expected,
        "location": location or {},
        "business_code": None if passed else code.value,
    }


def _truth_alignment(truth: dict[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    defects = truth.get("defects")
    if not isinstance(defects, list):
        return False, {"defects_is_list": False}, {"defects_is_list": True}
    counts = Counter(str(record.get("type")) for record in defects if isinstance(record, dict))
    declared_counts = truth.get("counts_by_type", {})
    expected_removed: set[int] = set()
    valid_entries = True
    for record in defects:
        if not isinstance(record, dict):
            valid_entries = False
            continue
        faces = record.get("faces", [])
        vertices = record.get("vertices", [])
        if not isinstance(faces, list) or not isinstance(vertices, list):
            valid_entries = False
            continue
        try:
            expected_removed.update(int(value) for value in faces)
        except (TypeError, ValueError):
            valid_entries = False
    observed = {
        "injection_count": truth.get("injection_count"),
        "defect_record_count": truth.get("defect_record_count"),
        "actual_record_count": len(defects),
        "counts_by_type": dict(sorted(counts.items())),
        "declared_counts_by_type": declared_counts,
        "expected_removed_faces": sorted(expected_removed),
        "valid_record_entries": valid_entries,
    }
    expected = {
        "injection_count": len(defects),
        "defect_record_count": len(defects),
        "counts_by_type": dict(sorted(counts.items())),
        "expected_removed_faces": truth.get("expected_removed_faces"),
        "valid_record_entries": True,
    }
    passed = (
        valid_entries
        and truth.get("injection_count") == len(defects)
        and truth.get("defect_record_count") == len(defects)
        and declared_counts == dict(sorted(counts.items()))
        and sorted(int(value) for value in truth.get("expected_removed_faces", []))
        == sorted(expected_removed)
    )
    return passed, observed, expected


def validate_files(mesh_path: str | Path, truth_path: str | Path) -> dict[str, Any]:
    mesh_target = Path(mesh_path)
    truth_target = Path(truth_path)
    checks: list[dict[str, Any]] = []
    try:
        truth = json.loads(truth_target.read_text(encoding="utf-8"))
        truth_loaded = isinstance(truth, dict)
        truth_error = None
    except (OSError, json.JSONDecodeError) as exc:
        truth = {}
        truth_loaded = False
        truth_error = str(exc)

    try:
        vertices, faces = read_ply(mesh_target)
        mesh_loaded = True
        mesh_error = None
    except BusinessError as exc:
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
        mesh_loaded = False
        mesh_error = exc.message

    format_ok = mesh_loaded and truth_loaded
    checks.append(
        _check(
            "mesh_format_valid",
            format_ok,
            criterion="strict_binary_little_endian_triangle_ply_and_json_manifest",
            observed={"mesh_loaded": mesh_loaded, "truth_loaded": truth_loaded},
            expected={"mesh_loaded": True, "truth_loaded": True},
            location={} if format_ok else {"criterion": "file_format"},
            code=ErrorCode.E_MESH_FORMAT,
        )
    )

    if format_ok:
        try:
            baseline = truth["baseline"]
            tolerances = truth["tolerances"]
            baseline_volume = float(baseline["volume"])
            baseline_area = float(baseline["area"])
            baseline_components = int(baseline["component_count"])
            baseline_face_count = int(baseline["face_count"])
            volume_tolerance = float(tolerances["volume_relative"])
            area_tolerance = float(tolerances["area_relative"])
            intersection_eps = float(tolerances["self_intersection_epsilon"])
            truth_fields_ok = all(
                np.isfinite(value)
                for value in (
                    baseline_volume,
                    baseline_area,
                    volume_tolerance,
                    area_tolerance,
                    intersection_eps,
                )
            )
        except (KeyError, TypeError, ValueError):
            truth_fields_ok = False
            baseline_volume = baseline_area = 0.0
            baseline_components = baseline_face_count = 0
            volume_tolerance = area_tolerance = 0.0
            intersection_eps = 1.0e-10

        edge_map = _python_edges(faces)
        nonmanifold = [(edge, values) for edge, values in edge_map.items() if len(values) > 2]
        boundary = [(edge, values) for edge, values in edge_map.items() if len(values) < 2]
        inconsistent = [
            (edge, values)
            for edge, values in edge_map.items()
            if len(values) == 2 and values[0][1] == values[1][1]
        ]
        labels, component_count = _python_components(faces)
        self_pairs = _python_self_intersections(vertices, faces, intersection_eps)
        triangles = vertices[faces] if len(faces) else np.empty((0, 3, 3))
        volume = (
            float(
                np.sum(
                    np.einsum(
                        "ij,ij->i",
                        triangles[:, 0],
                        np.cross(triangles[:, 1], triangles[:, 2]),
                    )
                )
                / 6.0
            )
            if len(faces)
            else 0.0
        )
        area = float(np.sum(triangle_areas(vertices, faces))) if len(faces) else 0.0
        volume_relative = abs(volume - baseline_volume) / max(abs(baseline_volume), 1.0e-15)
        area_relative = abs(area - baseline_area) / max(abs(baseline_area), 1.0e-15)
        truth_aligned, truth_observed, truth_expected = _truth_alignment(truth)

        checks.extend(
            [
                _check(
                    "nonmanifold_edges_zero",
                    not nonmanifold,
                    criterion="edge_incidence_gt_2",
                    observed=len(nonmanifold),
                    expected=0,
                    location=(
                        {"edge": list(nonmanifold[0][0]), "faces": [item[0] for item in nonmanifold[0][1]], "criterion": "edge_incidence_gt_2"}
                        if nonmanifold
                        else {}
                    ),
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "boundary_edges_zero",
                    not boundary,
                    criterion="closed_surface_boundary_incidence_eq_1",
                    observed=len(boundary),
                    expected=0,
                    location=(
                        {"edge": list(boundary[0][0]), "faces": [item[0] for item in boundary[0][1]], "criterion": "boundary_edge"}
                        if boundary
                        else {}
                    ),
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "component_count_matches_baseline",
                    component_count == baseline_components,
                    criterion="connected_component_count_eq_baseline",
                    observed=component_count,
                    expected=baseline_components,
                    location={} if component_count == baseline_components else {"component_count": component_count},
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "face_normals_consistent",
                    not inconsistent,
                    criterion="shared_edge_windings_opposite",
                    observed=len(inconsistent),
                    expected=0,
                    location=(
                        {"edge": list(inconsistent[0][0]), "faces": [item[0] for item in inconsistent[0][1]], "criterion": "winding_conflict"}
                        if inconsistent
                        else {}
                    ),
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "self_intersections_zero",
                    not self_pairs,
                    criterion="triangle_triangle_intersection_without_shared_vertex",
                    observed=len(self_pairs),
                    expected=0,
                    location={"faces": list(self_pairs[0]), "criterion": "triangle_intersection"} if self_pairs else {},
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "volume_within_tolerance",
                    truth_fields_ok and volume_relative <= volume_tolerance,
                    criterion="abs_volume_delta_over_baseline_le_tolerance",
                    observed={"volume": volume, "relative_deviation": volume_relative},
                    expected={"baseline_volume": baseline_volume, "tolerance": volume_tolerance},
                    location={} if truth_fields_ok and volume_relative <= volume_tolerance else {"criterion": "volume_relative"},
                    code=ErrorCode.E_VOLUME_DEVIATE,
                ),
                _check(
                    "area_within_tolerance",
                    truth_fields_ok and area_relative <= area_tolerance,
                    criterion="abs_area_delta_over_baseline_le_tolerance",
                    observed={"area": area, "relative_deviation": area_relative},
                    expected={"baseline_area": baseline_area, "tolerance": area_tolerance},
                    location={} if truth_fields_ok and area_relative <= area_tolerance else {"criterion": "area_relative"},
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "face_count_matches_baseline",
                    len(faces) == baseline_face_count,
                    criterion="retained_face_count_eq_baseline_face_count",
                    observed=len(faces),
                    expected=baseline_face_count,
                    location={} if len(faces) == baseline_face_count else {"criterion": "face_count"},
                    code=ErrorCode.E_RESIDUAL_TOPOLOGY,
                ),
                _check(
                    "truth_defects_aligned",
                    truth_aligned,
                    criterion="ground_truth_counts_and_removed_face_regions_are_consistent",
                    observed=truth_observed,
                    expected=truth_expected,
                    location={} if truth_aligned else {"criterion": "truth_alignment"},
                    code=ErrorCode.E_MESH_FORMAT,
                ),
            ]
        )
        metrics = {
            "volume": volume,
            "baseline_volume": baseline_volume,
            "volume_relative_deviation": volume_relative,
            "area": area,
            "baseline_area": baseline_area,
            "area_relative_deviation": area_relative,
            "component_count": component_count,
            "baseline_component_count": baseline_components,
            "face_count": len(faces),
            "baseline_face_count": baseline_face_count,
            "nonmanifold_edge_count": len(nonmanifold),
            "boundary_edge_count": len(boundary),
            "inconsistent_winding_edge_count": len(inconsistent),
            "self_intersection_count": len(self_pairs),
            "vertex_component_count_labels": len(set(labels)),
        }
    else:
        metrics = {
            "mesh_error": mesh_error,
            "truth_error": truth_error,
        }

    failed = next((check for check in checks if check["status"] == "fail"), None)
    passed = failed is None
    business_code = "OK" if passed else str(failed["business_code"])

    return {
        "schema_version": 1,
        "verifier": "independent_file_level_triangle_mesh",
        "status": "pass" if passed else "fail",
        "business_code": business_code,
        "inputs": {
            "mesh_file": mesh_target.name,
            "truth_file": truth_target.name,
            "mesh_sha256": sha256_file(mesh_target) if mesh_target.exists() else None,
            "truth_sha256": sha256_file(truth_target) if truth_target.exists() else None,
        },
        "checks": checks,
        "checks_by_name": {check["name"]: check for check in checks},
        "first_failure": failed,
        "metrics": metrics,
    }
