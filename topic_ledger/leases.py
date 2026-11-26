from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BusinessError, E_LEASE_EXPIRED, E_LEASE_GENERATION


def monotonic_ms() -> int:
    return int(time.monotonic_ns() // 1_000_000)


@dataclass
class SlotTicket:
    semaphore: threading.BoundedSemaphore | None
    node_id: str
    global_pool: "SlotPool | None" = None
    released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def release(self) -> bool:
        with self._lock:
            if self.released:
                return False
            self.released = True
        if self.semaphore is not None:
            self.semaphore.release()
        if self.global_pool is not None:
            self.global_pool._release_node(self.node_id)
            self.global_pool._release_global()
        return True


class SlotPool:
    def __init__(self, global_concurrency: int, node_quotas: dict[str, int]) -> None:
        self.global_limit = global_concurrency
        self.global_slots = threading.BoundedSemaphore(global_concurrency)
        self.node_slots = {
            node_id: threading.BoundedSemaphore(quota) for node_id, quota in node_quotas.items()
        }
        self._global_held = 0
        self._node_held: dict[str, int] = {node_id: 0 for node_id in node_quotas}
        self._lock = threading.Lock()

    def acquire(self, node_id: str, *, timeout_seconds: float | None = None) -> SlotTicket:
        node_semaphore = self.node_slots[node_id]
        acquired_global = self.global_slots.acquire(timeout=timeout_seconds) if timeout_seconds is not None else self.global_slots.acquire()
        if not acquired_global:
            raise TimeoutError("global concurrency slot unavailable")
        acquired_node = node_semaphore.acquire(timeout=timeout_seconds) if timeout_seconds is not None else node_semaphore.acquire()
        if not acquired_node:
            self.global_slots.release()
            raise TimeoutError(f"node concurrency slot unavailable: {node_id}")
        with self._lock:
            self._global_held += 1
            self._node_held[node_id] = self._node_held.get(node_id, 0) + 1
        return SlotTicket(semaphore=node_semaphore, node_id=node_id, global_pool=self)

    def _release_global(self) -> None:
        self.global_slots.release()
        with self._lock:
            self._global_held = max(0, self._global_held - 1)

    def _release_node(self, node_id: str) -> None:
        with self._lock:
            self._node_held[node_id] = max(0, self._node_held.get(node_id, 0) - 1)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global_held": self._global_held,
                "global_limit": self.global_limit,
                "node_held": dict(sorted(self._node_held.items())),
            }


@dataclass
class Lease:
    shard_id: str
    node_id: str
    generation: int
    issued_at_ms: int
    expires_at_ms: int
    ttl_ms: int
    batch_id: str
    status: str = "active"
    ticket: SlotTicket | None = field(default=None, repr=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "generation": self.generation,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "ttl_ms": self.ttl_ms,
            "batch_id": self.batch_id,
            "status": self.status,
        }


class LeaseManager:
    def __init__(
        self,
        *,
        ttl_ms: int,
        clock: Callable[[], int] = monotonic_ms,
    ) -> None:
        self.ttl_ms = ttl_ms
        self.clock = clock
        self.generation = 0
        self.leases: dict[str, Lease] = {}
        self.ended: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def _event(self, event: str, lease: Lease, code: str | None = None) -> None:
        item = {"event": event, "code": code, **lease.public_dict()}
        self.events.append(item)

    def active_count(self, node_id: str) -> int:
        with self._lock:
            return sum(
                1
                for lease in self.leases.values()
                if lease.node_id == node_id and lease.status == "active"
            )

    def active_generation(self, shard_id: str) -> int | None:
        with self._lock:
            lease = self.leases.get(shard_id)
            return lease.generation if lease and lease.status == "active" else None

    def is_current(self, shard_id: str, generation: int) -> bool:
        return self.active_generation(shard_id) == generation

    def attach_ticket(self, shard_id: str, generation: int, ticket: SlotTicket) -> bool:
        with self._lock:
            lease = self.leases.get(shard_id)
            if lease is None or lease.status != "active" or lease.generation != generation:
                ticket.release()
                return False
            lease.ticket = ticket
            return True

    def issue(
        self,
        *,
        shard_id: str,
        node_id: str,
        lease_limit: int,
        batch_id: str,
        ttl_ms: int | None = None,
        ticket: SlotTicket | None = None,
    ) -> Lease:
        with self._lock:
            existing = self.leases.get(shard_id)
            if existing and existing.status == "active":
                raise BusinessError(
                    E_LEASE_GENERATION,
                    "shard already has an active lease",
                    field="shard_id",
                    batch_id=batch_id,
                    details={"current_generation": existing.generation},
                )
            if self.active_count(node_id) >= lease_limit:
                raise BusinessError(
                    "E_NODE_LEASE_LIMIT",
                    "node lease limit reached",
                    field="lease_limit",
                    batch_id=batch_id,
                    details={"node_id": node_id, "lease_limit": lease_limit},
                )
            now = self.clock()
            ttl = self.ttl_ms if ttl_ms is None else ttl_ms
            self.generation += 1
            lease = Lease(
                shard_id=shard_id,
                node_id=node_id,
                generation=self.generation,
                issued_at_ms=now,
                expires_at_ms=now + ttl,
                ttl_ms=ttl,
                batch_id=batch_id,
                ticket=ticket,
            )
            self.leases[shard_id] = lease
            self._event("issued", lease)
            return lease

    def renew(self, *, shard_id: str, generation: int, ttl_ms: int | None = None) -> Lease:
        with self._lock:
            lease = self.leases.get(shard_id)
            if lease is None or lease.status != "active":
                raise BusinessError(
                    E_LEASE_GENERATION,
                    "no active lease for shard",
                    field="generation",
                    details={"shard_id": shard_id, "supplied_generation": generation},
                )
            if lease.generation != generation:
                raise BusinessError(
                    E_LEASE_GENERATION,
                    "lease generation mismatch; generation was not advanced",
                    field="generation",
                    batch_id=lease.batch_id,
                    details={
                        "shard_id": shard_id,
                        "supplied_generation": generation,
                        "current_generation": lease.generation,
                    },
                )
            lease.expires_at_ms = self.clock() + (self.ttl_ms if ttl_ms is None else ttl_ms)
            self._event("renewed", lease)
            return lease

    def release(self, *, shard_id: str, generation: int) -> Lease:
        with self._lock:
            lease = self.leases.get(shard_id)
            if lease is None or lease.status != "active":
                raise BusinessError(
                    E_LEASE_GENERATION,
                    "no active lease to release",
                    field="generation",
                    details={"shard_id": shard_id, "supplied_generation": generation},
                )
            if lease.generation != generation:
                raise BusinessError(
                    E_LEASE_GENERATION,
                    "release generation mismatch; generation was not advanced",
                    field="generation",
                    batch_id=lease.batch_id,
                    details={
                        "supplied_generation": generation,
                        "current_generation": lease.generation,
                    },
                )
            lease.status = "released"
            if lease.ticket is not None:
                lease.ticket.release()
            self.ended.append(lease.public_dict() | {"end_status": "released"})
            self._event("released", lease)
            return lease

    def recover_expired(self, now_ms: int | None = None) -> list[Lease]:
        with self._lock:
            now = self.clock() if now_ms is None else now_ms
            expired: list[Lease] = []
            for shard_id in sorted(self.leases):
                lease = self.leases[shard_id]
                if lease.status == "active" and now >= lease.expires_at_ms:
                    lease.status = "expired"
                    if lease.ticket is not None:
                        lease.ticket.release()
                    self.ended.append(
                        lease.public_dict() | {"end_status": "expired", "code": E_LEASE_EXPIRED}
                    )
                    self._event("expired", lease, E_LEASE_EXPIRED)
                    expired.append(lease)
            return expired

    def active_report(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.public_dict() for shard, lease in sorted(self.leases.items()) if lease.status == "active"]

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_generation": self.generation,
                "active": self.active_report(),
                "ended": list(self.ended),
                "events": list(self.events),
            }
