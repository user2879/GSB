"""Repair planning, bounded parallel block execution and deterministic merge."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .boundary import BoundaryResolution, VertexProposal, resolve_boundary_proposals
from .errors import BusinessError, ErrorCode
from .mesh import (
    Mesh,
    edge_data,
    edge_diagnostics,
    find_self_intersections,
    mesh_components,
    outward_flip_mask,
    triangle_areas,
)
from .preprocessing import ChunkManifest, RepairBlock


ZERO_AREA_EPS = 1.0e-14


@dataclass(frozen=True)
class RepairPlan:
    delete_reasons: dict[int, str]
    flip_faces: tuple[int, ...]
    zero_area_faces: tuple[int, ...]
    self_intersection_faces: tuple[int, ...]
    nonmanifold_faces: tuple[int, ...]
    isolated_component_faces: tuple[int, ...]
    active_face_ids: tuple[int, ...]
    self_intersection_pairs: tuple[tuple[int, int], ...]
    initial_nonmanifold_edge_count: int
    resolved_nonmanifold_edges: tuple[dict[str, Any], ...]
    component_count_before: int
    component_count_after: int

    def to_dict(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for reason in self.delete_reasons.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "delete_count": len(self.delete_reasons),
            "delete_count_by_reason": reasons,
            "flip_count": len(self.flip_faces),
            "zero_area_faces": list(self.zero_area_faces),
            "self_intersection_faces": list(self.self_intersection_faces),
            "self_intersection_pairs": [list(pair) for pair in self.self_intersection_pairs],
            "nonmanifold_faces": list(self.nonmanifold_faces),
            "isolated_component_faces": list(self.isolated_component_faces),
            "active_face_count": len(self.active_face_ids),
            "initial_nonmanifold_edge_count": self.initial_nonmanifold_edge_count,
            "resolved_nonmanifold_edge_count": len(self.resolved_nonmanifold_edges),
            "resolved_nonmanifold_edges": list(self.resolved_nonmanifold_edges),
            "component_count_before": self.component_count_before,
            "component_count_after": self.component_count_after,
        }


def _component_component_count(faces: np.ndarray) -> int:
    _, count = mesh_components(faces)
    return count


def build_repair_plan(
    mesh: Mesh,
    *,
    isolated_component_face_threshold: int,
    intersection_epsilon: float,
) -> RepairPlan:
    faces = mesh.faces
    areas = triangle_areas(mesh.vertices, faces)
    reasons: dict[int, str] = {}
    zero_area = tuple(int(index) for index in np.nonzero(areas <= ZERO_AREA_EPS)[0])
    for face in zero_area:
        reasons[face] = "zero_area_face"

    self_pairs = tuple(
        (int(first), int(second))
        for first, second in find_self_intersections(
            mesh.vertices, faces, eps=intersection_epsilon
        )
        if first not in reasons and second not in reasons
    )
    self_faces: set[int] = set()
    for first, second in self_pairs:
        self_faces.update((first, second))
        reasons.setdefault(first, "self_intersection")
        reasons.setdefault(second, "self_intersection")

    component_before = _component_component_count(faces)
    initial_nonmanifold = edge_diagnostics(faces)["nonmanifold_edge_count"]
    active = np.ones(len(faces), dtype=bool)
    active[np.fromiter(reasons.keys(), dtype=np.int64, count=len(reasons))] = False
    nonmanifold_deleted: set[int] = set()
    resolved_edges: list[dict[str, Any]] = []

    for _ in range(len(faces) + 1):
        active_ids = np.nonzero(active)[0].astype(np.int64)
        if len(active_ids) == 0:
            break
        local_faces = faces[active_ids]
        local_edges = edge_data(local_faces)
        violations = [
            edge_index
            for edge_index, incident in enumerate(local_edges.incident_faces)
            if len(incident) > 2
        ]
        if not violations:
            break
        to_delete: set[int] = set()
        for edge_index in violations:
            incident = list(local_edges.incident_faces[edge_index])
            boundary_counts: dict[int, int] = {}
            for local_face in incident:
                boundary_counts[local_face] = sum(
                    1
                    for candidate_edge in local_edges.face_edge_indices(local_face)
                    if len(local_edges.incident_faces[candidate_edge]) == 1
                )
            ranked = sorted(
                incident,
                key=lambda local_face: (
                    boundary_counts[local_face],
                    int(active_ids[local_face]),
                ),
                reverse=True,
            )
            remove_count = max(0, len(incident) - 2)
            for local_face in ranked[:remove_count]:
                global_face = int(active_ids[local_face])
                to_delete.add(global_face)
                resolved_edges.append(
                    {
                        "edge": [
                            int(local_edges.edges[edge_index][0]),
                            int(local_edges.edges[edge_index][1]),
                        ],
                        "incidence": len(incident),
                        "removed_face": global_face,
                        "rank_criterion": "boundary_edge_count_then_face_id_desc",
                    }
                )
        if not to_delete:
            raise BusinessError(
                ErrorCode.E_RESIDUAL_TOPOLOGY,
                "non-manifold edge could not be resolved",
                location={"edge_count": len(violations)},
            )
        for face in sorted(to_delete):
            reasons.setdefault(face, "non_manifold_edge")
            nonmanifold_deleted.add(face)
            active[face] = False

    active_ids = np.nonzero(active)[0].astype(np.int64)
    local_faces = faces[active_ids]
    labels, local_component_count = mesh_components(local_faces)
    isolated_faces: set[int] = set()
    for component in range(local_component_count):
        local_members = np.nonzero(labels == component)[0]
        if len(local_members) < isolated_component_face_threshold:
            for local_face in local_members:
                global_face = int(active_ids[int(local_face)])
                reasons.setdefault(global_face, "isolated_component")
                isolated_faces.add(global_face)
                active[global_face] = False

    active_ids = np.nonzero(active)[0].astype(np.int64)
    local_faces = faces[active_ids]
    if len(local_faces) == 0:
        raise BusinessError(ErrorCode.E_RESIDUAL_TOPOLOGY, "repair plan removed every face")
    try:
        local_flips = outward_flip_mask(mesh.vertices, local_faces)
    except ValueError as exc:
        raise BusinessError(
            ErrorCode.E_RESIDUAL_TOPOLOGY,
            "active faces cannot be oriented consistently",
            details={"error": str(exc)},
        ) from exc
    flip_faces = tuple(int(active_ids[index]) for index in np.nonzero(local_flips)[0])

    active_diagnostics = edge_diagnostics(local_faces)
    if (
        active_diagnostics["nonmanifold_edge_count"]
        or active_diagnostics["boundary_edge_count"]
        or active_diagnostics["inconsistent_winding_edge_count"]
    ):
        raise BusinessError(
            ErrorCode.E_RESIDUAL_TOPOLOGY,
            "repair plan still contains invalid topology",
            details=active_diagnostics,
        )
    component_after = _component_component_count(local_faces)
    return RepairPlan(
        delete_reasons=dict(sorted(reasons.items())),
        flip_faces=tuple(sorted(flip_faces)),
        zero_area_faces=tuple(sorted(zero_area)),
        self_intersection_faces=tuple(sorted(self_faces)),
        nonmanifold_faces=tuple(sorted(nonmanifold_deleted)),
        isolated_component_faces=tuple(sorted(isolated_faces)),
        active_face_ids=tuple(int(value) for value in active_ids),
        self_intersection_pairs=self_pairs,
        initial_nonmanifold_edge_count=int(initial_nonmanifold),
        resolved_nonmanifold_edges=tuple(resolved_edges),
        component_count_before=component_before,
        component_count_after=component_after,
    )


def estimate_block_memory_bytes(block: RepairBlock) -> int:
    return int(4096 + 96 * len(block.vertex_indices) + 192 * len(block.face_indices))


def estimate_block_time_ms(block: RepairBlock, threads: int) -> float:
    work = (len(block.face_indices) + 0.5 * len(block.vertex_indices)) / 1000.0
    contention = 1.0 + 0.05 * max(0, threads - 1)
    return float(0.02 + work * contention)


@dataclass(frozen=True)
class SchedulerReport:
    requested_threads: int
    effective_threads: int
    block_count: int
    max_block_memory_bytes: int
    peak_memory_required_bytes: int
    max_estimated_block_ms: float
    memory_budget_bytes: int
    block_time_budget_ms: float
    fallback_steps: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_threads": self.requested_threads,
            "effective_threads": self.effective_threads,
            "block_count": self.block_count,
            "max_block_memory_bytes": self.max_block_memory_bytes,
            "peak_memory_required_bytes": self.peak_memory_required_bytes,
            "max_estimated_block_ms": self.max_estimated_block_ms,
            "memory_budget_bytes": self.memory_budget_bytes,
            "block_time_budget_ms": self.block_time_budget_ms,
            "fallback_steps": list(self.fallback_steps),
            "degraded": bool(self.fallback_steps),
        }


@dataclass(frozen=True)
class BlockWorkResult:
    key: str
    retained_faces: tuple[tuple[int, tuple[int, int, int]], ...]
    deleted_faces: tuple[dict[str, Any], ...]
    flipped_faces: tuple[dict[str, Any], ...]
    proposals: tuple[VertexProposal, ...]

    def summary(self, volume: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "face_count": volume,
            "retained_face_count": len(self.retained_faces),
            "deleted_face_count": len(self.deleted_faces),
            "flipped_face_count": len(self.flipped_faces),
            "boundary_proposal_count": len(self.proposals),
        }


@dataclass(frozen=True)
class RepairResult:
    mesh: Mesh
    plan: RepairPlan
    actions: dict[str, Any]
    boundary_resolution: BoundaryResolution
    scheduler_report: SchedulerReport
    block_summaries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "scheduler": self.scheduler_report.to_dict(),
            "boundary": self.boundary_resolution.to_dict(),
            "blocks": list(self.block_summaries),
            "actions": self.actions,
        }


def _apply_block(
    block: RepairBlock,
    mesh: Mesh,
    plan: RepairPlan,
) -> BlockWorkResult:
    retained: list[tuple[int, tuple[int, int, int]]] = []
    deleted: list[dict[str, Any]] = []
    flipped: list[dict[str, Any]] = []
    for face_id in block.face_indices:
        original = tuple(int(value) for value in mesh.faces[face_id])
        reason = plan.delete_reasons.get(face_id)
        if reason is not None:
            deleted.append(
                {
                    "face": face_id,
                    "vertices": list(original),
                    "reason": reason,
                }
            )
            continue
        if face_id in plan.flip_faces:
            output = (original[0], original[2], original[1])
            flipped.append(
                {
                    "face": face_id,
                    "original_winding": list(original),
                    "output_winding": list(output),
                    "reason": "consistent_outward_orientation",
                }
            )
        else:
            output = original
        retained.append((face_id, output))

    retained_vertices = {vertex for _face_id, face in retained for vertex in face}
    proposals = tuple(
        VertexProposal(
            block_key=block.key,
            vertex=vertex,
            position=tuple(float(value) for value in mesh.vertices[vertex]),
            keep=vertex in retained_vertices,
        )
        for vertex in block.boundary_vertices
    )
    return BlockWorkResult(
        key=block.key,
        retained_faces=tuple(retained),
        deleted_faces=tuple(deleted),
        flipped_faces=tuple(flipped),
        proposals=proposals,
    )


def _choose_budget(
    manifest: ChunkManifest,
    *,
    requested_threads: int,
    memory_budget_bytes: int,
    block_time_budget_ms: float,
) -> SchedulerReport:
    blocks = manifest.blocks
    if not blocks:
        raise BusinessError(ErrorCode.E_BUDGET_DENIED, "there are no repair blocks to schedule")
    max_memory = max(estimate_block_memory_bytes(block) for block in blocks)
    workers = min(max(1, requested_threads), len(blocks))
    fallback: list[dict[str, Any]] = []

    while workers > 1:
        max_time = max(estimate_block_time_ms(block, workers) for block in blocks)
        memory_required = workers * max_memory
        time_failure = max_time > block_time_budget_ms
        memory_failure = memory_required > memory_budget_bytes
        if not time_failure and not memory_failure:
            break
        reason = "time_budget" if time_failure else "memory_budget"
        fallback.append(
            {
                "from_threads": workers,
                "to_threads": workers - 1,
                "reason": reason,
                "max_estimated_block_ms": max_time,
                "peak_memory_required_bytes": memory_required,
                "priority": "time_budget_then_memory_budget",
            }
        )
        workers -= 1

    max_time = max(estimate_block_time_ms(block, workers) for block in blocks)
    memory_required = workers * max_memory
    if max_time > block_time_budget_ms:
        raise BusinessError(
            ErrorCode.E_BUDGET_DENIED,
            "single-block time budget is insufficient even at serial execution",
            details={
                "estimated_block_ms": max_time,
                "budget_ms": block_time_budget_ms,
                "effective_threads": workers,
            },
        )
    if memory_required > memory_budget_bytes:
        raise BusinessError(
            ErrorCode.E_BUDGET_DENIED,
            "memory budget cannot hold the largest repair block",
            details={
                "required_bytes": memory_required,
                "budget_bytes": memory_budget_bytes,
                "largest_block_bytes": max_memory,
            },
        )
    return SchedulerReport(
        requested_threads=requested_threads,
        effective_threads=workers,
        block_count=len(blocks),
        max_block_memory_bytes=max_memory,
        peak_memory_required_bytes=memory_required,
        max_estimated_block_ms=max_time,
        memory_budget_bytes=memory_budget_bytes,
        block_time_budget_ms=block_time_budget_ms,
        fallback_steps=tuple(fallback),
    )


def execute_repair(
    mesh: Mesh,
    manifest: ChunkManifest,
    plan: RepairPlan,
    *,
    requested_threads: int,
    memory_budget_bytes: int,
    block_time_budget_ms: float,
    boundary_tolerance: float,
) -> RepairResult:
    scheduler_report = _choose_budget(
        manifest,
        requested_threads=requested_threads,
        memory_budget_bytes=memory_budget_bytes,
        block_time_budget_ms=block_time_budget_ms,
    )
    results: dict[str, BlockWorkResult] = {}
    with ThreadPoolExecutor(max_workers=scheduler_report.effective_threads) as executor:
        futures: dict[str, Future[BlockWorkResult]] = {
            block.key: executor.submit(_apply_block, block, mesh, plan)
            for block in manifest.blocks
        }
        for block in manifest.blocks:
            future = futures[block.key]
            try:
                results[block.key] = future.result()
            except BusinessError:
                raise
            except Exception as exc:
                raise BusinessError(
                    ErrorCode.E_INTERNAL,
                    "repair worker failed",
                    location={"block": block.key},
                    details={"error": repr(exc)},
                ) from exc

    expected_keys = [block.key for block in manifest.blocks]
    if sorted(results) != sorted(expected_keys):
        missing = sorted(set(expected_keys).difference(results))
        raise BusinessError(
            ErrorCode.E_BLOCK_KEY,
            "repair result set is incomplete",
            details={"missing_blocks": missing},
        )

    ordered_results = [results[key] for key in sorted(results)]
    retained_entries: list[tuple[int, tuple[int, int, int]]] = []
    deleted_entries: list[dict[str, Any]] = []
    flipped_entries: list[dict[str, Any]] = []
    proposals: list[VertexProposal] = []
    for result in ordered_results:
        retained_entries.extend(result.retained_faces)
        deleted_entries.extend(result.deleted_faces)
        flipped_entries.extend(result.flipped_faces)
        proposals.extend(result.proposals)

    retained_entries.sort(key=lambda entry: entry[0])
    deleted_entries.sort(key=lambda entry: int(entry["face"]))
    flipped_entries.sort(key=lambda entry: int(entry["face"]))
    retained_ids = [face_id for face_id, _face in retained_entries]
    if len(retained_ids) != len(set(retained_ids)):
        raise BusinessError(ErrorCode.E_BLOCK_KEY, "a retained face was merged more than once")
    expected_active = list(plan.active_face_ids)
    if retained_ids != expected_active:
        raise BusinessError(
            ErrorCode.E_BLOCK_KEY,
            "merged retained faces do not match the repair plan",
            details={
                "expected_count": len(expected_active),
                "actual_count": len(retained_ids),
            },
        )

    resolution = resolve_boundary_proposals(
        proposals,
        tolerance=boundary_tolerance,
        vertex_count=mesh.vertex_count,
    )
    used_vertices = sorted(
        {vertex for _face_id, face in retained_entries for vertex in face}
    )
    for vertex in used_vertices:
        if vertex in resolution.keep and not resolution.keep[vertex]:
            raise BusinessError(
                ErrorCode.E_BOUNDARY_CONFLICT,
                "boundary arbiter dropped a vertex required by a retained face",
                location={"vertex": vertex},
            )

    remap = {vertex: index for index, vertex in enumerate(used_vertices)}
    final_vertices = np.asarray(
        [
            resolution.positions.get(
                vertex,
                tuple(float(value) for value in mesh.vertices[vertex]),
            )
            for vertex in used_vertices
        ],
        dtype=np.float64,
    )
    final_faces = np.asarray(
        [[remap[vertex] for vertex in face] for _face_id, face in retained_entries],
        dtype=np.int64,
    )
    final_mesh = Mesh(vertices=final_vertices, faces=final_faces)

    reason_counts: dict[str, int] = {}
    for entry in deleted_entries:
        reason = str(entry["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    actions = {
        "deleted_faces": deleted_entries,
        "flipped_faces": flipped_entries,
        "deleted_count_by_reason": dict(sorted(reason_counts.items())),
        "deleted_face_count": len(deleted_entries),
        "flipped_face_count": len(flipped_entries),
        "output_face_count": final_mesh.face_count,
        "output_vertex_count": final_mesh.vertex_count,
    }
    block_summaries = tuple(
        {
            **result.summary(len(next(block for block in manifest.blocks if block.key == result.key).face_indices)),
            "estimated_memory_bytes": estimate_block_memory_bytes(
                next(block for block in manifest.blocks if block.key == result.key)
            ),
            "estimated_time_ms": estimate_block_time_ms(
                next(block for block in manifest.blocks if block.key == result.key),
                scheduler_report.effective_threads,
            ),
        }
        for result in ordered_results
    )
    return RepairResult(
        mesh=final_mesh,
        plan=plan,
        actions=actions,
        boundary_resolution=resolution,
        scheduler_report=scheduler_report,
        block_summaries=block_summaries,
    )
