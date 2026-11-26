from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .content import expected_shard
from .spec import ArtifactSpec, NodeSpec


ARBITRATION_RULE = "digest_match_desc:free_slots_desc:node_id_asc:v1"


@dataclass(frozen=True)
class CandidateAssessment:
    node_id: str
    digest_match: bool
    free_slots_before: int
    active_leases_before: int
    max_leases: int
    capacity_eligible: bool
    selected: bool
    veto_reasons: tuple[str, ...]
    sort_key: tuple[int, int, str]
    shards: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "digest_match": self.digest_match,
            "free_slots_before": self.free_slots_before,
            "active_leases_before": self.active_leases_before,
            "max_leases": self.max_leases,
            "capacity_eligible": self.capacity_eligible,
            "selected": self.selected,
            "veto_reasons": list(self.veto_reasons),
            "sort_key": list(self.sort_key),
            "shards": list(self.shards),
        }


@dataclass(frozen=True)
class Arbitration:
    artifact_key: str
    owner_node_id: str | None
    required_slots: int
    rule: str
    candidates: tuple[CandidateAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_key": self.artifact_key,
            "owner_node_id": self.owner_node_id,
            "required_slots": self.required_slots,
            "rule": self.rule,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _digest_rows(
    node: NodeSpec,
    artifact: ArtifactSpec,
    node_observations: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], bool, list[int]]:
    rows: list[dict[str, Any]] = []
    mismatches: list[int] = []
    all_match = True
    for shard in artifact.shards:
        _content, digest = expected_shard(node, artifact.artifact_key, shard)
        observed_digest = None if node_observations is None else node_observations.get(shard.index)
        match = observed_digest is None or observed_digest == digest
        if not match:
            all_match = False
            mismatches.append(shard.index)
        rows.append(
            {
                "shard_index": shard.index,
                "derived_digest": digest,
                "observed_digest": observed_digest,
                "match": match,
            }
        )
    return rows, all_match, mismatches


def arbitrate_ownership(
    *,
    artifact: ArtifactSpec,
    nodes: list[NodeSpec] | tuple[NodeSpec, ...],
    free_slots: dict[str, int],
    active_lease_counts: dict[str, int],
    observed_digests: dict[str, dict[int, str]] | None = None,
    pinned_node_id: str | None = None,
) -> Arbitration:
    required_slots = len(artifact.shards)
    candidates: list[CandidateAssessment] = []
    observations = observed_digests or {}

    for node in sorted(nodes, key=lambda item: item.node_id):
        node_observations = observations.get(node.node_id)
        rows, digest_match, mismatches = _digest_rows(node, artifact, node_observations)

        free = int(free_slots.get(node.node_id, node.capacity_slots))
        active_leases = int(active_lease_counts.get(node.node_id, 0))
        reasons: list[str] = []
        if mismatches:
            reasons.append("digest_mismatch:" + ",".join(map(str, mismatches)))
        if free < required_slots:
            reasons.append("insufficient_capacity")
        if active_leases >= node.max_leases:
            reasons.append("lease_limit_reached")
        capacity_eligible = free >= required_slots and active_leases < node.max_leases
        if pinned_node_id is not None and node.node_id != pinned_node_id:
            reasons.append("pinned_other_node")
        if pinned_node_id == node.node_id and not capacity_eligible:
            reasons.append("pinned_node_without_capacity")

        sort_key = (0 if digest_match else 1, -free, node.node_id)
        candidates.append(
            CandidateAssessment(
                node_id=node.node_id,
                digest_match=digest_match,
                free_slots_before=free,
                active_leases_before=active_leases,
                max_leases=node.max_leases,
                capacity_eligible=capacity_eligible,
                selected=False,
                veto_reasons=tuple(reasons),
                sort_key=sort_key,
                shards=tuple(rows),
            )
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate.capacity_eligible
        and not any(reason.startswith("pinned_other_node") for reason in candidate.veto_reasons)
    ]
    if pinned_node_id is not None:
        pinned = [candidate for candidate in eligible if candidate.node_id == pinned_node_id]
        winner = pinned[0] if pinned else None
    else:
        winner = min(eligible, key=lambda candidate: candidate.sort_key) if eligible else None

    assessed: list[CandidateAssessment] = []
    for candidate in candidates:
        reasons = list(candidate.veto_reasons)
        selected = winner is not None and candidate.node_id == winner.node_id
        if not selected and winner is not None and candidate.capacity_eligible:
            if not candidate.digest_match and winner.digest_match:
                reasons.append("digest_match_priority_lost")
            elif candidate.free_slots_before < winner.free_slots_before:
                reasons.append("free_slots_desc_lost")
            elif candidate.node_id > winner.node_id:
                reasons.append("node_id_order_lost")
            else:
                reasons.append("arbitration_order_lost")
        assessed.append(
            CandidateAssessment(
                node_id=candidate.node_id,
                digest_match=candidate.digest_match,
                free_slots_before=candidate.free_slots_before,
                active_leases_before=candidate.active_leases_before,
                max_leases=candidate.max_leases,
                capacity_eligible=candidate.capacity_eligible,
                selected=selected,
                veto_reasons=tuple(reasons),
                sort_key=candidate.sort_key,
                shards=candidate.shards,
            )
        )

    return Arbitration(
        artifact_key=artifact.artifact_key,
        owner_node_id=None if winner is None else winner.node_id,
        required_slots=required_slots,
        rule=ARBITRATION_RULE,
        candidates=tuple(assessed),
    )


def assign_lowest_free_slots(occupied_slots: set[int], capacity_slots: int, count: int) -> list[int]:
    free = [slot for slot in range(capacity_slots) if slot not in occupied_slots]
    if len(free) < count:
        raise ValueError("not enough free capacity slots")
    return free[:count]
