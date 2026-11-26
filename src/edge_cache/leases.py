"""Centre-side lease bookkeeping: expiry, watermarks and capacity slots.

The centre never *fakes* reclamation with sleeps or thread timeouts.  A lease
holds an integer capacity slot on its node and one global concurrency token
for its whole lifetime.  :meth:`LeaseManager.reclaim_expired` is the single
place where both resources are released back; it runs before every admission
decision and after every scheduler tick, and every reclamation emits an
``E_LEASE_EXPIRED`` ledger failure plus a node algebra event (a tagged
release through the node protocol).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, SystemClock
from .config import NodeConfig
from .errors import (
    E_BUDGET_EXHAUSTED,
    E_HARD_WATERMARK,
    E_LEASE_EXPIRED,
    E_LEASE_LIMIT,
    Failure,
)

SOFT_WATERMARK = "SOFT_WATERMARK"
HARD_WATERMARK = "HARD_WATERMARK"


@dataclass
class Lease:
    lease_id: str
    node_id: str
    shard_id: str
    tag: int
    issued_ms: int
    expires_ms: int
    slots: int
    renewed_count: int = 0


@dataclass
class WatermarkEvent:
    node_id: str
    level: str
    occupied: int
    capacity: int
    at_ms: int


@dataclass
class LeaseManager:
    nodes: dict[str, NodeConfig]
    global_concurrency: int
    clock: Clock = field(default_factory=SystemClock)
    leases: dict[str, Lease] = field(default_factory=dict)
    watermarks: list[WatermarkEvent] = field(default_factory=list)
    soft_open: set[str] = field(default_factory=set)
    expiry_events: list[Failure] = field(default_factory=list)
    reclamation_callback: Any = None

    def active_lease_count(self, node_id: str | None = None) -> int:
        if node_id is None:
            return len(self.leases)
        return sum(1 for lease in self.leases.values() if lease.node_id == node_id)

    def occupied_slots(self, node_id: str) -> int:
        return sum(lease.slots for lease in self.leases.values() if lease.node_id == node_id)

    def reclaim_expired(self, now_ms: int) -> list[tuple[Lease, Failure]]:
        """Really release every lease whose deadline has passed."""

        reclaimed: list[tuple[Lease, Failure]] = []
        for lease_id in sorted(self.leases):
            lease = self.leases[lease_id]
            if now_ms < lease.expires_ms:
                continue
            if self.reclamation_callback is not None:
                # The callback performs the tagged release on the node so the
                # node's generation algebra advances on real reclamation.
                self.reclamation_callback(lease, now_ms)
            failure = Failure(
                E_LEASE_EXPIRED,
                "lease expired before renewal and was reclaimed",
                {"lease_id": lease.lease_id, "issued_ms": lease.issued_ms,
                 "expires_ms": lease.expires_ms, "at_ms": now_ms},
                shard_id=lease.shard_id,
                node_id=lease.node_id,
            )
            self.expiry_events.append(failure)
            reclaimed.append((lease, failure))
            del self.leases[lease_id]
        return reclaimed

    def check_admission(
        self, node: NodeConfig, slots: int, now_ms: int, budget_remaining: int
    ) -> Failure | None:
        if budget_remaining <= 0:
            return Failure(E_BUDGET_EXHAUSTED, "batch point budget exhausted",
                           {"at_ms": now_ms}, node_id=node.node_id)
        if len(self.leases) >= self.global_concurrency:
            return Failure(E_LEASE_LIMIT, "global concurrency limit reached",
                           {"global_concurrency": self.global_concurrency, "at_ms": now_ms},
                           node_id=node.node_id)
        if self.active_lease_count(node.node_id) >= node.lease_limit:
            return Failure(E_LEASE_LIMIT, "per-node lease limit reached",
                           {"lease_limit": node.lease_limit, "at_ms": now_ms},
                           node_id=node.node_id)
        occupied = self.occupied_slots(node.node_id)
        if occupied + slots > node.capacity_slots:
            return Failure(E_HARD_WATERMARK, "integer capacity slots exhausted",
                           {"occupied": occupied, "requested": slots,
                            "capacity": node.capacity_slots, "at_ms": now_ms},
                           node_id=node.node_id)
        projected = occupied + slots
        soft = node.capacity_slots * node.soft_watermark_permille // 1000
        hard = node.capacity_slots * node.hard_watermark_permille // 1000
        if projected > hard:
            return Failure(E_HARD_WATERMARK, "hard watermark crossed",
                           {"projected": projected, "hard": hard, "at_ms": now_ms},
                           node_id=node.node_id)
        if projected > soft and node.node_id not in self.soft_open:
            self.soft_open.add(node.node_id)
            self.watermarks.append(
                WatermarkEvent(node.node_id, SOFT_WATERMARK, projected,
                               node.capacity_slots, now_ms)
            )
        return None

    def register(self, lease: Lease) -> None:
        self.leases[lease.lease_id] = lease

    def remove(self, lease_id: str) -> Lease | None:
        return self.leases.pop(lease_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "leases": [lease.__dict__ for lease in sorted(self.leases.values(),
                                                          key=lambda item: item.lease_id)],
            "watermarks": [event.__dict__ for event in self.watermarks],
            "soft_open": sorted(self.soft_open),
            "expiry_events": [failure.to_dict() for failure in self.expiry_events],
        }
