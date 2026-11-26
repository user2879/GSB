"""Triple quota and budget accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .spec import NodeSpec, ShardSpec


@dataclass
class Allocation:
    shard_id: str
    node_id: str
    units: int
    lease_id: str
    admitted_ns: int
    released: bool = False
    release_reason: str | None = None


class CapacityState:
    def __init__(self, budget: int, nodes: list[NodeSpec] | tuple[NodeSpec, ...]) -> None:
        self.condition = threading.Condition(threading.RLock())
        self.budget = int(budget)
        self.node_quota = {node.id: int(node.concurrency) for node in nodes}
        self.lease_limit = {node.id: int(node.lease_limit) for node in nodes}
        self.allocations: dict[str, Allocation] = {}
        self.active_units = 0
        self.active_shards_by_node = {node.id: 0 for node in nodes}
        self.active_leases_by_node = {node.id: 0 for node in nodes}
        self.peak_units = 0
        self.peak_shards_by_node = {node.id: 0 for node in nodes}
        self.peak_leases_by_node = {node.id: 0 for node in nodes}

    def update_budget(self, budget: int) -> None:
        with self.condition:
            self.budget = int(budget)
            self.condition.notify_all()

    def update_quota(self, node_id: str, concurrency: int) -> None:
        with self.condition:
            self.node_quota[node_id] = int(concurrency)
            self.condition.notify_all()

    def update_lease_limit(self, node_id: str, lease_limit: int) -> None:
        with self.condition:
            self.lease_limit[node_id] = int(lease_limit)
            self.condition.notify_all()

    def check(self, shard: ShardSpec, *, active_leases: int | None = None) -> dict[str, Any]:
        with self.condition:
            active_shards = self.active_shards_by_node[shard.node_id]
            leases = self.active_leases_by_node[shard.node_id] if active_leases is None else active_leases
            quota = self.node_quota[shard.node_id]
            lease_limit = self.lease_limit[shard.node_id]
            reasons = []
            if shard.units > 0 and self.active_units + shard.units > self.budget:
                reasons.append("global_budget")
            if active_shards + 1 > quota:
                reasons.append("node_concurrency")
            if leases + 1 > lease_limit:
                reasons.append("node_lease_limit")
            return {
                "admittable": not reasons,
                "reasons": reasons,
                "active_units": self.active_units,
                "budget": self.budget,
                "remaining_units": self.budget - self.active_units,
                "active_shards": active_shards,
                "concurrency": quota,
                "active_leases": leases,
                "lease_limit": lease_limit,
            }

    def admit(self, shard: ShardSpec, lease_id: str, now_ns: int) -> Allocation | None:
        with self.condition:
            check = self.check(shard)
            if not check["admittable"]:
                return None
            allocation = Allocation(
                shard_id=shard.id,
                node_id=shard.node_id,
                units=shard.units,
                lease_id=lease_id,
                admitted_ns=now_ns,
            )
            self.allocations[shard.id] = allocation
            self.active_units += shard.units
            self.active_shards_by_node[shard.node_id] += 1
            self.active_leases_by_node[shard.node_id] += 1
            self.peak_units = max(self.peak_units, self.active_units)
            self.peak_shards_by_node[shard.node_id] = max(
                self.peak_shards_by_node[shard.node_id], self.active_shards_by_node[shard.node_id]
            )
            self.peak_leases_by_node[shard.node_id] = max(
                self.peak_leases_by_node[shard.node_id], self.active_leases_by_node[shard.node_id]
            )
            self.condition.notify_all()
            return allocation

    def release(self, shard_id: str, reason: str = "completed") -> Allocation | None:
        with self.condition:
            allocation = self.allocations.pop(shard_id, None)
            if allocation is None or allocation.released:
                return allocation
            allocation.released = True
            allocation.release_reason = reason
            self.active_units = max(0, self.active_units - allocation.units)
            self.active_shards_by_node[allocation.node_id] = max(
                0, self.active_shards_by_node[allocation.node_id] - 1
            )
            self.active_leases_by_node[allocation.node_id] = max(
                0, self.active_leases_by_node[allocation.node_id] - 1
            )
            self.condition.notify_all()
            return allocation

    def reclaim_all(self, reason: str = "crash_recovery") -> list[Allocation]:
        with self.condition:
            allocations = list(self.allocations.values())
            for allocation in allocations:
                if not allocation.released:
                    allocation.released = True
                    allocation.release_reason = reason
            self.allocations.clear()
            self.active_units = 0
            for node_id in self.active_shards_by_node:
                self.active_shards_by_node[node_id] = 0
                self.active_leases_by_node[node_id] = 0
            self.condition.notify_all()
            return allocations

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "budget": self.budget,
                "active_units": self.active_units,
                "remaining_units": self.budget - self.active_units,
                "node_quota": dict(self.node_quota),
                "lease_limit": dict(self.lease_limit),
                "active_shards_by_node": dict(self.active_shards_by_node),
                "active_leases_by_node": dict(self.active_leases_by_node),
                "peak_units": self.peak_units,
                "peak_shards_by_node": dict(self.peak_shards_by_node),
                "peak_leases_by_node": dict(self.peak_leases_by_node),
                "in_flight": sorted(self.allocations),
            }
