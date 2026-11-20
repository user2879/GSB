"""Incremental-versus-full mesh comparison with first-divergence context."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .errors import ErrorCode, StudioError
from .mesh import CombinedMesh, PatchMesh


def _distribution(deviations: np.ndarray, tolerance: float) -> dict[str, Any]:
    if deviations.size == 0:
        return {
            "sample_count": 0,
            "maximum": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "thresholds": {
                "le_1e-12": 0,
                "le_1e-9": 0,
                "le_1e-6": 0,
                "le_tolerance": 0,
                "gt_tolerance": 0,
            },
        }
    return {
        "sample_count": int(deviations.size),
        "maximum": float(deviations.max()),
        "mean": float(deviations.mean()),
        "p50": float(np.percentile(deviations, 50)),
        "p95": float(np.percentile(deviations, 95)),
        "p99": float(np.percentile(deviations, 99)),
        "thresholds": {
            "le_1e-12": int(np.count_nonzero(deviations <= 1.0e-12)),
            "le_1e-9": int(np.count_nonzero(deviations <= 1.0e-9)),
            "le_1e-6": int(np.count_nonzero(deviations <= 1.0e-6)),
            "le_tolerance": int(np.count_nonzero(deviations <= tolerance)),
            "gt_tolerance": int(np.count_nonzero(deviations > tolerance)),
        },
    }


def compare_meshes(
    incremental: CombinedMesh,
    full: CombinedMesh,
    patches: Mapping[str, PatchMesh],
    *,
    tolerance: float,
    edit_seq: int,
    injected_vertex_offset: tuple[float, float, float] | None = None,
    raise_on_mismatch: bool = True,
) -> dict[str, Any]:
    incremental_keys = set(incremental.patch_keys)
    full_keys = set(full.patch_keys)
    key_difference = incremental_keys != full_keys
    first_key: str | None = None
    first_local_vertex = -1
    first_global_vertex = -1
    first_deviation: float | None = None
    deviation_chunks: list[np.ndarray] = []
    ledger_violations = 0
    topology_mismatch = False

    all_keys = sorted(incremental_keys | full_keys, key=lambda item: item.encode("ascii"))
    for key in all_keys:
        if key not in incremental_keys or key not in full_keys:
            if first_key is None:
                first_key = key
            continue
        inc_start, inc_end = incremental.patch_vertex_offsets[key]
        full_start, full_end = full.patch_vertex_offsets[key]
        inc_vertices = incremental.vertices[inc_start:inc_end].copy()
        full_vertices = full.vertices[full_start:full_end].copy()
        if injected_vertex_offset is not None and inc_vertices.shape[0]:
            inc_vertices[0] += np.asarray(injected_vertex_offset, dtype=np.float64)
        if inc_vertices.shape != full_vertices.shape:
            topology_mismatch = True
            if first_key is None:
                first_key = key
            continue
        inc_triangles = incremental.triangles[
            np.all(incremental.triangles >= inc_start, axis=1)
            & np.all(incremental.triangles < inc_end, axis=1)
        ]
        full_triangles = full.triangles[
            np.all(full.triangles >= full_start, axis=1)
            & np.all(full.triangles < full_end, axis=1)
        ]
        if inc_triangles.shape != full_triangles.shape or not np.array_equal(
            inc_triangles - inc_start, full_triangles - full_start
        ):
            topology_mismatch = True
            if first_key is None:
                first_key = key
        if inc_vertices.shape[0]:
            deviation = np.linalg.norm(inc_vertices - full_vertices, axis=1)
            deviation_chunks.append(deviation)
            nonzero = np.flatnonzero(deviation > 0.0)
            local = int(nonzero[0]) if nonzero.size else -1
            if first_key is None and local >= 0:
                first_key = key
                first_local_vertex = local
                first_global_vertex = inc_start + local
                first_deviation = float(deviation[local])
            ledger_bound = patches[key].position_error_upper_bound + tolerance
            ledger_violations += int(np.count_nonzero(deviation > ledger_bound))

    deviations = np.concatenate(deviation_chunks) if deviation_chunks else np.zeros(0, dtype=np.float64)
    distribution = _distribution(deviations, tolerance)
    maximum = float(distribution["maximum"])
    within_tolerance = not key_difference and not topology_mismatch and maximum <= tolerance
    within_ledger_bound = ledger_violations == 0
    consistent = within_tolerance and within_ledger_bound
    if first_key is None and not consistent:
        first_key = all_keys[0] if all_keys else None
    result = {
        "ok": consistent,
        "edit_seq": int(edit_seq),
        "tolerance": float(tolerance),
        "criteria": {
            "vertex_position_within_tolerance": within_tolerance,
            "deviation_within_ledger_upper_bound": within_ledger_bound,
            "topology_equal": not topology_mismatch,
            "patch_key_sets_equal": not key_difference,
        },
        "maximum_position_deviation": maximum,
        "deviation_distribution": distribution,
        "ledger_violation_vertex_count": int(ledger_violations),
        "first_divergence": {
            "edit_seq": int(edit_seq),
            "patch_key": first_key,
            "local_vertex_index": first_local_vertex,
            "global_vertex_index": first_global_vertex,
            "position_deviation": first_deviation,
        },
        "incremental_mesh_sha256": incremental.content_sha256,
        "full_mesh_sha256": full.content_sha256,
    }
    if raise_on_mismatch and not consistent:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "incremental mesh is inconsistent with full reconstruction",
            context=result,
        )
    return result
