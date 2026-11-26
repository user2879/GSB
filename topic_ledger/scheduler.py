from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import E_BUDGET_PENDING, E_NODE_LEASE_LIMIT


@dataclass
class PendingShard:
    shard_id: str
    reason: str
    batch_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "reason": self.reason,
            "batch_id": self.batch_id,
            "detail": dict(sorted(self.detail.items())),
        }


@dataclass
class ScheduleItem:
    shard_id: str
    node_id: str
    cost: int
    batch_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "cost": self.cost,
            "batch_id": self.batch_id,
        }


@dataclass
class ScheduleResult:
    items: list[ScheduleItem]
    pending: list[PendingShard]
    consumed_budget: int
    budget_limit: int
    node_blocked: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "pending": [item.as_dict() for item in self.pending],
            "consumed_budget": self.consumed_budget,
            "remaining_budget": max(0, self.budget_limit - self.consumed_budget),
            "budget_limit": self.budget_limit,
            "node_blocked": dict(sorted(self.node_blocked.items())),
        }


def build_schedule(
    *,
    ordered_shards: list[str],
    routing: dict[str, str],
    costs: dict[str, int],
    batch_ids: dict[str, str],
    budget_limit: int,
    lease_limits: dict[str, int],
    initial_node_lease_counts: dict[str, int] | None = None,
    existing_pending: list[PendingShard] | None = None,
) -> ScheduleResult:
    previous_pending = list(existing_pending or [])
    pending_ids = {item.shard_id for item in previous_pending}
    # Existing pending work is always considered first on the next round.
    candidates = [item.shard_id for item in previous_pending]
    for shard_id in ordered_shards:
        if shard_id not in pending_ids:
            candidates.append(shard_id)

    projected_leases = dict(initial_node_lease_counts or {})
    node_blocked: dict[str, str] = {}
    items: list[ScheduleItem] = []
    pending: list[PendingShard] = []
    consumed = 0
    budget_exhausted = False

    for shard_id in candidates:
        node_id = routing[shard_id]
        batch_id = batch_ids[shard_id]
        cost = int(costs.get(shard_id, 0))

        if node_id in node_blocked or projected_leases.get(node_id, 0) >= lease_limits[node_id]:
            node_blocked[node_id] = E_NODE_LEASE_LIMIT
            pending.append(
                PendingShard(
                    shard_id=shard_id,
                    reason=E_NODE_LEASE_LIMIT,
                    batch_id=batch_id,
                    detail={"node_id": node_id, "lease_limit": lease_limits[node_id]},
                )
            )
            continue

        if budget_exhausted or cost > max(0, budget_limit - consumed):
            budget_exhausted = True
            pending.append(
                PendingShard(
                    shard_id=shard_id,
                    reason=E_BUDGET_PENDING,
                    batch_id=batch_id,
                    detail={
                        "cost": cost,
                        "remaining_budget": max(0, budget_limit - consumed),
                    },
                )
            )
            continue

        items.append(ScheduleItem(shard_id=shard_id, node_id=node_id, cost=cost, batch_id=batch_id))
        consumed += cost
        projected_leases[node_id] = projected_leases.get(node_id, 0) + 1

    return ScheduleResult(
        items=items,
        pending=pending,
        consumed_budget=consumed,
        budget_limit=budget_limit,
        node_blocked=node_blocked,
    )
