"""Pipeline-side quality checks and ground-truth action alignment."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .errors import ErrorCode
from .mesh import (
    Mesh,
    edge_diagnostics,
    find_self_intersections,
    mesh_components,
    signed_volume,
    surface_area,
)


COMMON_CHECK_NAMES = (
    "nonmanifold_edges_zero",
    "boundary_edges_zero",
    "component_count_matches_baseline",
    "face_normals_consistent",
    "self_intersections_zero",
    "volume_within_tolerance",
    "area_within_tolerance",
    "face_count_matches_baseline",
    "truth_defects_aligned",
)


def _check(
    name: str,
    passed: bool,
    *,
    criterion: str,
    observed: Any,
    expected: Any,
    location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "criterion": criterion,
        "observed": observed,
        "expected": expected,
        "location": location or {},
    }


def _record_alignment(
    truth: dict[str, Any],
    deleted_by_face: dict[int, str],
    weld_report: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]], list[int]]:
    removed_vertices = set(int(value) for value in weld_report.get("removed_vertices", []))
    results: list[dict[str, Any]] = []
    all_passed = True
    legitimate_faces_deleted: list[int] = []
    baseline_face_count = int(truth.get("baseline", {}).get("face_count", 0))
    for face, reason in deleted_by_face.items():
        if face < baseline_face_count:
            legitimate_faces_deleted.append(face)

    expected_reason = {
        "zero_area_face": "zero_area_face",
        "self_intersection": "self_intersection",
        "non_manifold_edge": "non_manifold_edge",
        "isolated_component": "isolated_component",
    }
    for record in truth.get("defects", []):
        defect_type = str(record.get("type"))
        faces = [int(value) for value in record.get("faces", [])]
        vertices = [int(value) for value in record.get("vertices", [])]
        if defect_type == "duplicate_vertex":
            passed = bool(vertices) and all(vertex in removed_vertices for vertex in vertices)
            observed = {
                "vertices": vertices,
                "welded_removed_vertices": sorted(removed_vertices.intersection(vertices)),
            }
        else:
            reason = expected_reason.get(defect_type)
            observed = {
                "faces": faces,
                "observed_reasons": {str(face): deleted_by_face.get(face, "retained") for face in faces},
            }
            passed = reason is not None and bool(faces) and all(
                deleted_by_face.get(face) == reason for face in faces
            )
        all_passed = all_passed and passed
        results.append(
            {
                "defect_id": record.get("defect_id"),
                "type": defect_type,
                "processed": passed,
                "observed": observed,
            }
        )
    return all_passed, results, sorted(legitimate_faces_deleted)


def _triangle_coordinate_keys(mesh: Mesh) -> Counter[tuple[tuple[float, float, float], ...]]:
    keys: Counter[tuple[tuple[float, float, float], ...]] = Counter()
    for face in mesh.faces:
        points = [tuple(float(value) for value in mesh.vertices[vertex]) for vertex in face]
        rotations = [tuple(points), tuple(points[1:] + points[:1]), tuple(points[2:] + points[:2])]
        keys[min(rotations)] += 1
    return keys


def compute_quality(
    repaired: Mesh,
    baseline: Mesh,
    truth: dict[str, Any],
    *,
    weld_report: dict[str, Any],
    actions: dict[str, Any],
    intersection_epsilon: float,
) -> dict[str, Any]:
    diagnostics = edge_diagnostics(repaired.faces)
    _, component_count = mesh_components(repaired.faces)
    self_pairs = find_self_intersections(
        repaired.vertices, repaired.faces, eps=intersection_epsilon
    )
    volume = signed_volume(repaired.vertices, repaired.faces)
    area = surface_area(repaired.vertices, repaired.faces)
    baseline_truth = truth["baseline"]
    baseline_volume = float(baseline_truth["volume"])
    baseline_area = float(baseline_truth["area"])
    baseline_components = int(baseline_truth["component_count"])
    baseline_face_count = int(baseline_truth["face_count"])
    volume_tolerance = float(truth["tolerances"]["volume_relative"])
    area_tolerance = float(truth["tolerances"]["area_relative"])
    volume_relative = abs(volume - baseline_volume) / max(abs(baseline_volume), 1.0e-15)
    area_relative = abs(area - baseline_area) / max(abs(baseline_area), 1.0e-15)

    deleted_by_face = {
        int(entry["face"]): str(entry["reason"])
        for entry in actions.get("deleted_faces", [])
    }
    alignment_passed, alignment, misdeleted_faces = _record_alignment(
        truth, deleted_by_face, weld_report
    )
    expected_removed = set(int(value) for value in truth.get("expected_removed_faces", []))
    observed_removed = set(deleted_by_face)
    alignment_passed = alignment_passed and expected_removed == observed_removed
    if expected_removed != observed_removed:
        misdeleted_faces = sorted(
            set(misdeleted_faces)
            | observed_removed.difference(expected_removed)
        )

    legal_geometry_preserved = _triangle_coordinate_keys(repaired) == _triangle_coordinate_keys(baseline)
    checks = [
        _check(
            "nonmanifold_edges_zero",
            diagnostics["nonmanifold_edge_count"] == 0,
            criterion="edge_incidence_gt_2",
            observed=diagnostics["nonmanifold_edge_count"],
            expected=0,
            location=(
                {"edge": diagnostics["first_nonmanifold_edge"], "criterion": "edge_incidence_gt_2"}
                if diagnostics["first_nonmanifold_edge"]
                else {}
            ),
        ),
        _check(
            "boundary_edges_zero",
            diagnostics["boundary_edge_count"] == 0,
            criterion="closed_surface_boundary_incidence_eq_1",
            observed=diagnostics["boundary_edge_count"],
            expected=0,
            location=(
                {"edge": diagnostics["first_boundary_edge"], "criterion": "boundary_edge"}
                if diagnostics["first_boundary_edge"]
                else {}
            ),
        ),
        _check(
            "component_count_matches_baseline",
            component_count == baseline_components,
            criterion="connected_component_count_eq_baseline",
            observed=component_count,
            expected=baseline_components,
            location={} if component_count == baseline_components else {"component_count": component_count},
        ),
        _check(
            "face_normals_consistent",
            diagnostics["inconsistent_winding_edge_count"] == 0,
            criterion="shared_edge_windings_opposite",
            observed=diagnostics["inconsistent_winding_edge_count"],
            expected=0,
            location=(
                {
                    "face": diagnostics["first_inconsistent_face"],
                    "edge": diagnostics["first_inconsistent_edge"],
                    "criterion": "winding_conflict",
                }
                if diagnostics["first_inconsistent_face"] is not None
                else {}
            ),
        ),
        _check(
            "self_intersections_zero",
            len(self_pairs) == 0,
            criterion="triangle_triangle_intersection_without_shared_vertex",
            observed=len(self_pairs),
            expected=0,
            location=(
                {"faces": list(self_pairs[0]), "criterion": "triangle_intersection"}
                if self_pairs
                else {}
            ),
        ),
        _check(
            "volume_within_tolerance",
            volume_relative <= volume_tolerance,
            criterion="abs_volume_delta_over_baseline_le_tolerance",
            observed={"volume": volume, "relative_deviation": volume_relative},
            expected={"baseline_volume": baseline_volume, "tolerance": volume_tolerance},
            location={} if volume_relative <= volume_tolerance else {"criterion": "volume_relative"},
        ),
        _check(
            "area_within_tolerance",
            area_relative <= area_tolerance,
            criterion="abs_area_delta_over_baseline_le_tolerance",
            observed={"area": area, "relative_deviation": area_relative},
            expected={"baseline_area": baseline_area, "tolerance": area_tolerance},
            location={} if area_relative <= area_tolerance else {"criterion": "area_relative"},
        ),
        _check(
            "face_count_matches_baseline",
            repaired.face_count == baseline_face_count,
            criterion="retained_face_count_eq_baseline_face_count",
            observed=repaired.face_count,
            expected=baseline_face_count,
            location={} if repaired.face_count == baseline_face_count else {"criterion": "face_count"},
        ),
        _check(
            "truth_defects_aligned",
            alignment_passed,
            criterion="each_injected_defect_has_matching_repair_action",
            observed={"processed_records": alignment, "unexpected_removed_faces": sorted(observed_removed.difference(expected_removed))},
            expected={"expected_removed_faces": sorted(expected_removed)},
            location={} if alignment_passed else {"criterion": "truth_alignment"},
        ),
    ]
    common_pass = all(check["status"] == "pass" for check in checks)
    checks.append(
        _check(
            "legal_face_geometry_preserved",
            legal_geometry_preserved,
            criterion="baseline_triangle_coordinate_multiset_unchanged",
            observed={"face_count": repaired.face_count, "coordinate_multiset_equal": legal_geometry_preserved},
            expected={"face_count": baseline_face_count, "coordinate_multiset_equal": True},
            location={} if legal_geometry_preserved else {"criterion": "legal_face_geometry"},
        )
    )
    passed = common_pass and legal_geometry_preserved and not misdeleted_faces
    if not passed:
        topology_failed = any(
            check["name"] in COMMON_CHECK_NAMES
            and "volume" not in check["name"]
            and check["status"] == "fail"
            for check in checks
        ) or not legal_geometry_preserved or bool(misdeleted_faces)
        code = ErrorCode.E_RESIDUAL_TOPOLOGY if topology_failed else ErrorCode.E_VOLUME_DEVIATE
    else:
        code = None
    failed = next((check for check in checks if check["status"] == "fail"), None)
    return {
        "status": "pass" if passed else "fail",
        "business_code": code.value if code else "OK",
        "checks": checks,
        "checks_by_name": {check["name"]: check for check in checks},
        "first_failure": failed,
        "metrics": {
            "volume": volume,
            "baseline_volume": baseline_volume,
            "volume_relative_deviation": volume_relative,
            "area": area,
            "baseline_area": baseline_area,
            "area_relative_deviation": area_relative,
            "component_count": component_count,
            "baseline_component_count": baseline_components,
            "face_count": repaired.face_count,
            "baseline_face_count": baseline_face_count,
            "nonmanifold_edge_count": diagnostics["nonmanifold_edge_count"],
            "boundary_edge_count": diagnostics["boundary_edge_count"],
            "inconsistent_winding_edge_count": diagnostics["inconsistent_winding_edge_count"],
            "self_intersection_count": len(self_pairs),
        },
        "misdeleted_legal_faces": misdeleted_faces,
        "legal_face_geometry_preserved": legal_geometry_preserved,
        "defect_alignment": alignment,
    }
