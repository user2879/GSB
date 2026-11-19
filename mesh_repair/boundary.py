"""Deterministic single-writer arbitration for block-shared vertices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .errors import BusinessError, ErrorCode


@dataclass(frozen=True)
class VertexProposal:
    block_key: str
    vertex: int
    position: tuple[float, float, float]
    keep: bool


@dataclass(frozen=True)
class BoundaryConflict:
    vertex: int
    criterion: str
    resolved_by: str
    blocks: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "criterion": self.criterion,
            "resolved_by": self.resolved_by,
            "blocks": list(self.blocks),
            "positions": [list(position) for position in self.positions],
        }


@dataclass(frozen=True)
class BoundaryResolution:
    positions: dict[int, tuple[float, float, float]]
    keep: dict[int, bool]
    proposal_count: int
    shared_vertex_count: int
    position_conflicts: tuple[BoundaryConflict, ...]
    retention_conflicts: tuple[BoundaryConflict, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arbiter": "lexicographic_block_key_single_writer",
            "proposal_count": self.proposal_count,
            "shared_vertex_count": self.shared_vertex_count,
            "position_conflict_count": len(self.position_conflicts),
            "retention_conflict_count": len(self.retention_conflicts),
            "position_conflicts": [conflict.to_dict() for conflict in self.position_conflicts],
            "retention_conflicts": [conflict.to_dict() for conflict in self.retention_conflicts],
            "committed_vertex_count": sum(1 for value in self.keep.values() if value),
        }


def resolve_boundary_proposals(
    proposals: Iterable[VertexProposal], *, tolerance: float, vertex_count: int
) -> BoundaryResolution:
    grouped: dict[int, list[VertexProposal]] = {}
    seen: set[tuple[str, int]] = set()
    for proposal in proposals:
        if not proposal.block_key:
            raise BusinessError(ErrorCode.E_BOUNDARY_CONFLICT, "empty block key in proposal")
        if proposal.vertex < 0 or proposal.vertex >= vertex_count:
            raise BusinessError(
                ErrorCode.E_BOUNDARY_CONFLICT,
                "proposal vertex is outside the mesh",
                location={"block": proposal.block_key, "vertex": proposal.vertex},
            )
        if len(proposal.position) != 3 or not all(math.isfinite(value) for value in proposal.position):
            raise BusinessError(
                ErrorCode.E_BOUNDARY_CONFLICT,
                "proposal position is invalid",
                location={"block": proposal.block_key, "vertex": proposal.vertex},
            )
        key = (proposal.block_key, proposal.vertex)
        if key in seen:
            raise BusinessError(
                ErrorCode.E_BOUNDARY_CONFLICT,
                "a block registered the same boundary vertex twice",
                location={"block": proposal.block_key, "vertex": proposal.vertex},
            )
        seen.add(key)
        grouped.setdefault(proposal.vertex, []).append(proposal)

    positions: dict[int, tuple[float, float, float]] = {}
    keep: dict[int, bool] = {}
    position_conflicts: list[BoundaryConflict] = []
    retention_conflicts: list[BoundaryConflict] = []
    for vertex in sorted(grouped):
        ordered = sorted(grouped[vertex], key=lambda item: item.block_key)
        chosen = ordered[0]
        positions[vertex] = chosen.position
        keep[vertex] = any(item.keep for item in ordered)
        positions_array = np.asarray([item.position for item in ordered], dtype=np.float64)
        max_delta = 0.0
        if len(positions_array) > 1:
            deltas = np.linalg.norm(positions_array - positions_array[0], axis=1)
            max_delta = float(np.max(deltas))
        if max_delta > tolerance:
            position_conflicts.append(
                BoundaryConflict(
                    vertex=vertex,
                    criterion="position_delta_gt_boundary_tolerance",
                    resolved_by=chosen.block_key,
                    blocks=tuple(item.block_key for item in ordered),
                    positions=tuple(item.position for item in ordered),
                )
            )
        if any(item.keep for item in ordered) and any(not item.keep for item in ordered):
            retention_conflicts.append(
                BoundaryConflict(
                    vertex=vertex,
                    criterion="retention_disagreement_resolved_by_logical_or",
                    resolved_by=chosen.block_key,
                    blocks=tuple(item.block_key for item in ordered),
                    positions=tuple(item.position for item in ordered),
                )
            )

    return BoundaryResolution(
        positions=positions,
        keep=keep,
        proposal_count=sum(len(values) for values in grouped.values()),
        shared_vertex_count=len(grouped),
        position_conflicts=tuple(position_conflicts),
        retention_conflicts=tuple(retention_conflicts),
    )
