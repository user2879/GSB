from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .concurrency import BoundedOrderedEvaluator
from .fixedpoint import deviation_field, per_vertex_errors
from .mesh import MeshState
from .propagation import parallel_trace_sources
from .topology import TopologyData, build_topology, geometry_metrics, laplacian_curvature_metric


def parallel_degenerate_scan(
    state: MeshState,
    area_tolerance: float,
    concurrency: int,
    memory_budget_nodes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = state.fixed.astype(np.float64) / float(state.scale)
    faces = state.faces

    def check(face_index: int) -> bool:
        face = faces[face_index]
        if len(set(int(value) for value in face.tolist())) < 3:
            return True
        points = coordinates[face]
        cross = np.cross(points[1] - points[0], points[2] - points[0])
        return float(0.5 * np.linalg.norm(cross)) <= area_tolerance

    evaluator = BoundedOrderedEvaluator[bool](
        concurrency=concurrency,
        memory_budget_nodes=memory_budget_nodes,
    )
    tasks = [(index, lambda index=index: check(index), 4) for index in range(int(faces.shape[0]))]
    values = evaluator.map_indexed(tasks)
    return np.asarray(values, dtype=bool), evaluator.metrics.to_dict()


def analyze_state(
    state: MeshState,
    area_tolerance: float,
    concurrency: int,
    memory_budget_nodes: int,
) -> tuple[TopologyData, dict[str, Any]]:
    topology = build_topology(state.fixed, state.faces, state.scale, area_tolerance)
    parallel_mask, parallel_metrics = parallel_degenerate_scan(
        state, area_tolerance, concurrency, memory_budget_nodes
    )
    summary = topology.summary()
    summary["fixed_geometry"] = geometry_metrics(state.fixed, state.faces, state.scale)
    summary["parallel_face_scan"] = {
        **parallel_metrics,
        "mask_matches_indexed_pass": bool(np.array_equal(parallel_mask, topology.degenerate_mask)),
    }
    return topology, summary


def float_reference_geometry(state: MeshState) -> dict[str, object]:
    reference = state.reference
    faces = state.faces
    bbox_min = np.min(reference, axis=0)
    bbox_max = np.max(reference, axis=0)
    points = reference[faces]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    volume_numerator = float(
        np.sum(
            points[:, 0, 0]
            * (points[:, 1, 1] * points[:, 2, 2] - points[:, 1, 2] * points[:, 2, 1])
            + points[:, 0, 1]
            * (points[:, 1, 2] * points[:, 2, 0] - points[:, 1, 0] * points[:, 2, 2])
            + points[:, 0, 2]
            * (points[:, 1, 0] * points[:, 2, 1] - points[:, 1, 1] * points[:, 2, 0]),
            dtype=np.float64,
        )
    )
    return {
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_extent": [float(value) for value in (bbox_max - bbox_min)],
        "surface_area": float(np.sum(areas, dtype=np.float64)),
        "signed_volume": float(volume_numerator / 6.0),
        "mean_uniform_laplacian_magnitude": laplacian_curvature_metric(reference, faces),
        "role": "reference_only_not_used_for_verdict",
    }


def fixed_operator_effect(
    previous: MeshState | None,
    current: MeshState,
    current_geometry: dict[str, Any],
) -> dict[str, Any]:
    if previous is None:
        return {
            "basis": "fixed_point",
            "common_vertex_count": current.vertex_count,
            "added_vertex_count": 0,
            "removed_vertex_count": 0,
            "changed_vertex_count": 0,
            "changed_vertices": [],
            "max_abs_coordinate_delta": 0.0,
            "bbox_extent_delta": [0.0, 0.0, 0.0],
            "surface_area_delta": 0.0,
            "signed_volume_delta": 0.0,
            "curvature_delta": 0.0,
        }
    common = min(previous.vertex_count, current.vertex_count)
    if common:
        delta = (
            current.fixed[:common].astype(np.float64)
            - previous.fixed[:common].astype(np.float64)
        ) / float(current.scale)
        axis_max = np.max(np.abs(delta), axis=1)
        changed = [
            int(index)
            for index in np.flatnonzero(np.any(delta != 0.0, axis=1))
        ]
        maximum = float(np.max(axis_max)) if axis_max.size else 0.0
    else:
        changed = []
        maximum = 0.0
    previous_geometry = geometry_metrics(
        previous.fixed, previous.faces, previous.scale
    )
    bbox_delta = [
        float(current_geometry["bbox_extent"][axis])
        - float(previous_geometry["bbox_extent"][axis])
        for axis in range(3)
    ]
    return {
        "basis": "fixed_point",
        "common_vertex_count": common,
        "added_vertex_count": max(0, current.vertex_count - previous.vertex_count),
        "removed_vertex_count": max(0, previous.vertex_count - current.vertex_count),
        "changed_vertex_count": len(changed),
        "changed_vertices": changed,
        "max_abs_coordinate_delta": maximum,
        "bbox_extent_delta": bbox_delta,
        "surface_area_delta": float(current_geometry["surface_area"])
        - float(previous_geometry["surface_area"]),
        "signed_volume_delta": float(current_geometry["signed_volume"])
        - float(previous_geometry["signed_volume"]),
        "curvature_delta": float(
            current_geometry["mean_uniform_laplacian_magnitude"]
        )
        - float(previous_geometry["mean_uniform_laplacian_magnitude"]),
    }


def build_error_record(
    state: MeshState,
    tolerance: float,
    operator_index: int,
    direct_sources: Sequence[int],
) -> dict[str, Any]:
    table = per_vertex_errors(state.fixed, state.reference, state.scale, tolerance)
    exceeding = [record for record in table if bool(record["exceeds_tolerance"])]
    first = exceeding[0] if exceeding else None
    return {
        "operator_index": operator_index,
        "tolerance": tolerance,
        "fixed_point_scale": state.scale,
        "decision_basis": "fixed_point_coordinates",
        "float64_role": "reference_only",
        "direct_source_vertices": sorted(int(value) for value in direct_sources),
        "vertex_errors": table,
        "max_abs_error": float(max((float(row["max_abs_error"]) for row in table), default=0.0)),
        "exceeding_vertex_count": len(exceeding),
        "exceeding_vertices": [int(row["vertex"]) for row in exceeding],
        "first_tolerance_exceedance": (
            {
                "vertex": int(first["vertex"]),
                "operator_index": operator_index,
                "max_abs_error": float(first["max_abs_error"]),
                "tolerance": tolerance,
            }
            if first is not None
            else None
        ),
    }


def build_propagation_record(
    state: MeshState,
    topology: TopologyData,
    operator_index: int,
    sources: Sequence[int],
    trace_radius: int,
    tolerance: float,
    concurrency: int,
    memory_budget_nodes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deviations = deviation_field(state.fixed, state.reference, state.scale)
    return parallel_trace_sources(
        operator_index,
        sources,
        topology.vertex_neighbors,
        deviations,
        trace_radius,
        tolerance,
        concurrency,
        memory_budget_nodes,
    )


def diagnostics_business_codes(summary: dict[str, Any], stage: str) -> list[str]:
    codes: list[str] = []
    if int(summary.get("nonmanifold_edge_count", 0)) > 0:
        codes.append(f"W_{stage}_NONMANIFOLD_EDGE")
    if int(summary.get("boundary_edge_count", 0)) > 0:
        codes.append(f"W_{stage}_BOUNDARY_EDGE")
    if int(summary.get("degenerate_face_count", 0)) > 0:
        codes.append(f"W_{stage}_DEGENERATE_FACE")
    if int(summary.get("isolated_vertex_count", 0)) > 0:
        codes.append(f"W_{stage}_ISOLATED_VERTEX")
    if int(summary.get("orientation_conflict_edge_count", 0)) > 0:
        codes.append(f"W_{stage}_ORIENTATION_CONFLICT")
    if not bool(summary.get("rescan_matches", False)):
        codes.append(f"W_{stage}_RESCAN_MISMATCH")
    parallel = summary.get("parallel_face_scan", {})
    if parallel and not bool(parallel.get("mask_matches_indexed_pass", False)):
        codes.append(f"W_{stage}_PARALLEL_MASK_MISMATCH")
    return codes
