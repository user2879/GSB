from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeCapacity:
    node_id: str
    capacity_slots: int
    lease_limit: int
    used_slots: int = 0
    active_leases: int = 0

    @property
    def remaining_slots(self) -> int:
        return self.capacity_slots - self.used_slots

    @property
    def lease_remaining(self) -> int:
        return self.lease_limit - self.active_leases

    def watermark(self, scale: int) -> dict[str, Any]:
        if scale <= 0:
            raise ValueError("scale must be positive")
        fixed = (self.used_slots * scale) // self.capacity_slots
        return {
            "node_id": self.node_id,
            "used_slots": self.used_slots,
            "remaining_slots": self.remaining_slots,
            "capacity_slots": self.capacity_slots,
            "active_leases": self.active_leases,
            "lease_limit": self.lease_limit,
            "lease_remaining": self.lease_remaining,
            "watermark_fixed": fixed,
            "watermark_scale": scale,
            "truncation_direction": "floor_for_nonnegative_ratio_toward_zero",
            "quantization_error_bound_scaled": "strictly_less_than_1_integer_unit",
            "quantization_error_bound_ratio_numerator": 1,
            "quantization_error_bound_ratio_denominator": scale,
        }


@dataclass
class ClusterCapacity:
    nodes: dict[str, NodeCapacity]
    reservations: dict[str, str] = field(default_factory=dict)
    leases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, nodes: list[Any] | tuple[Any, ...]) -> "ClusterCapacity":
        return cls(
            nodes={
                node.node_id: NodeCapacity(
                    node_id=node.node_id,
                    capacity_slots=node.capacity_slots,
                    lease_limit=node.lease_limit,
                )
                for node in nodes
            }
        )

    def reserve(self, reservation_id: str, node_id: str) -> dict[str, Any]:
        if reservation_id in self.reservations:
            raise ValueError(f"reservation {reservation_id} already exists")
        capacity = self.nodes[node_id]
        if capacity.remaining_slots <= 0:
            raise OverflowError(f"node {node_id} has no remaining slots")
        capacity.used_slots += 1
        self.reservations[reservation_id] = node_id
        return capacity.watermark(1000)

    def rollback_reservation(self, reservation_id: str) -> dict[str, Any]:
        node_id = self.reservations.pop(reservation_id)
        capacity = self.nodes[node_id]
        if capacity.used_slots <= 0:
            raise AssertionError("slot counter cannot become negative")
        capacity.used_slots -= 1
        return capacity.watermark(1000)

    def grant(self, reservation_id: str, lease_id: str) -> str:
        node_id = self.reservations.pop(reservation_id)
        if lease_id in self.leases:
            raise ValueError(f"lease {lease_id} already tracked")
        capacity = self.nodes[node_id]
        if capacity.active_leases >= capacity.lease_limit:
            raise OverflowError(f"node {node_id} lease limit reached")
        capacity.active_leases += 1
        self.leases[lease_id] = node_id
        return node_id

    def release_lease(self, lease_id: str) -> str | None:
        node_id = self.leases.pop(lease_id, None)
        if node_id is None:
            return None
        capacity = self.nodes[node_id]
        capacity.active_leases -= 1
        capacity.used_slots -= 1
        return node_id


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
    return (
        0 if candidate.get("digest_match") else 1,
        -int(candidate["remaining_slots"]),
        str(candidate["node_id"]),
    )


def arbitrate_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the unique winner using digest, free slots, then node id."""
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["eligible"] = bool(item.get("eligible", True))
        item.setdefault("reasons", [])
        annotated.append(item)

    eligible = [item for item in annotated if item["eligible"]]
    if not eligible:
        return {
            "winner_node_id": None,
            "candidates": sorted(annotated, key=lambda item: str(item["node_id"])),
            "indistinguishable_node_ids": [],
            "basis": "no eligible candidate",
        }

    ranked = sorted(eligible, key=_candidate_key)
    winner = ranked[0]
    winner_rank_key = _candidate_key(winner)
    indistinguishable = [
        str(item["node_id"])
        for item in ranked
        if _candidate_key(item)[:2] == winner_rank_key[:2]
    ]

    for item in annotated:
        reasons = list(item.get("reasons", []))
        if not item.get("eligible", True):
            item["decision"] = "rejected"
            continue
        if item["node_id"] == winner["node_id"]:
            item["decision"] = "selected"
            if len(indistinguishable) > 1:
                reasons.append("STABLE_NODE_ID_TIE_BREAK")
        else:
            item["decision"] = "control"
            if not item.get("digest_match"):
                reasons.append("DIGEST_MISMATCH")
            elif int(item["remaining_slots"]) < int(winner["remaining_slots"]):
                reasons.append("LOWER_REMAINING_SLOTS")
            elif int(item["remaining_slots"]) == int(winner["remaining_slots"]):
                reasons.append("NODE_ID_TIE_BREAK")
            else:
                reasons.append("BEHIND_ON_STABLE_RANKING")
        item["reasons"] = reasons

    for item in annotated:
        item["indistinguishable"] = item["node_id"] in indistinguishable

    ordered = sorted(annotated, key=lambda item: (_candidate_key(item) if item.get("eligible") else (2, 0, str(item["node_id"]))))
    return {
        "winner_node_id": str(winner["node_id"]),
        "candidates": ordered,
        "indistinguishable_node_ids": sorted(indistinguishable),
        "basis": "digest_match_desc,remaining_slots_desc,node_id_asc",
    }
