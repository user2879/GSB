"""Monotonic-clock leases with optimistic versions and real expiry."""

from __future__ import annotations

import heapq
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .timeutil import now_ms, now_ns


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_at_ms: int
    expires_at_ms: int
    renew_count: int
    version: int
    capacity_version: int
    granted_ns: int
    expires_ns: int

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("granted_ns", None)
        data.pop("expires_ns", None)
        return data


class LeaseManager:
    def __init__(self, on_expire: Callable[[Lease], None] | None = None) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._leases: dict[str, Lease] = {}
        self._by_shard: dict[str, str] = {}
        self._heap: list[tuple[int, str]] = []
        self._on_expire = on_expire
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start_reaper(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reap_loop, name="shardmesh-lease-reaper", daemon=True)
        self._thread.start()

    def stop_reaper(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def issue(
        self,
        *,
        lease_id: str,
        shard_id: str,
        node_id: str,
        ttl_ms: int,
        capacity_version: int,
        version: int = 1,
        renew_count: int = 0,
    ) -> Lease:
        granted_ns = now_ns()
        ttl_ns = max(1, int(ttl_ms)) * 1_000_000
        lease = Lease(
            lease_id=lease_id,
            shard_id=shard_id,
            node_id=node_id,
            granted_at_ms=granted_ns // 1_000_000,
            expires_at_ms=(granted_ns + ttl_ns) // 1_000_000,
            renew_count=renew_count,
            version=version,
            capacity_version=capacity_version,
            granted_ns=granted_ns,
            expires_ns=granted_ns + ttl_ns,
        )
        with self._condition:
            self._leases[lease_id] = lease
            self._by_shard[shard_id] = lease_id
            heapq.heappush(self._heap, (lease.expires_ns, lease_id))
            self._condition.notify_all()
        return lease

    def get(self, lease_id: str) -> Lease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def get_for_shard(self, shard_id: str) -> Lease | None:
        with self._lock:
            lease_id = self._by_shard.get(shard_id)
            return self._leases.get(lease_id) if lease_id else None

    def active_count(self, node_id: str | None = None) -> int:
        current = now_ns()
        with self._lock:
            return sum(1 for lease in self._leases.values() if lease.expires_ns > current and (node_id is None or lease.node_id == node_id))

    def apply_renewal(self, lease_id: str, new_version: int, renew_count: int, ttl_ms: int) -> Lease:
        with self._condition:
            lease = self._leases[lease_id]
            lease.version = int(new_version)
            lease.renew_count = int(renew_count)
            lease.expires_ns = now_ns() + max(1, int(ttl_ms)) * 1_000_000
            lease.expires_at_ms = lease.expires_ns // 1_000_000
            heapq.heappush(self._heap, (lease.expires_ns, lease_id))
            self._condition.notify_all()
            return lease

    def remove(self, lease_id: str) -> Lease | None:
        with self._condition:
            lease = self._leases.pop(lease_id, None)
            if lease is not None:
                self._by_shard.pop(lease.shard_id, None)
            self._condition.notify_all()
            return lease

    def revoke_capacity_version(self, node_id: str, new_version: int) -> list[Lease]:
        revoked: list[Lease] = []
        with self._condition:
            for lease in list(self._leases.values()):
                if lease.node_id == node_id and lease.capacity_version < new_version:
                    removed = self._leases.pop(lease.lease_id, None)
                    self._by_shard.pop(lease.shard_id, None)
                    if removed is not None:
                        revoked.append(removed)
            self._condition.notify_all()
        return revoked

    def expire_due(self) -> list[Lease]:
        current = now_ns()
        expired: list[Lease] = []
        with self._condition:
            while self._heap and self._heap[0][0] <= current:
                _, lease_id = heapq.heappop(self._heap)
                lease = self._leases.get(lease_id)
                if lease is None or lease.expires_ns > current:
                    continue
                removed = self._leases.pop(lease_id, None)
                self._by_shard.pop(lease.shard_id, None)
                if removed is not None:
                    expired.append(removed)
        return expired

    def _reap_loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                timeout = 0.01
                if self._heap:
                    timeout = max(0.001, (self._heap[0][0] - now_ns()) / 1_000_000_000)
                self._condition.wait(timeout=min(timeout, 0.005))
            expired = self.expire_due()
            for lease in expired:
                if self._on_expire is not None:
                    self._on_expire(lease)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.public_dict() for lease in sorted(self._leases.values(), key=lambda item: item.lease_id)]


def monotonic_lease_age_ms(lease: Lease) -> int:
    return max(0, now_ms() - lease.granted_at_ms)
