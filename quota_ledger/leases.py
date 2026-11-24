"""Thread-safe leases with optimistic renewal and real monotonic expiry."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .client import NodeClient
from .errors import BusinessError
from .util import monotonic_ms


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_ms: int
    expires_ms: int
    version: int
    config_version: int
    generation: int
    ttl_ms: int
    state: str = "active"

    def public(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "version": self.version,
            "config_version": self.config_version,
            "generation": self.generation,
            "state": self.state,
        }


class LeaseManager:
    """One active lease per shard and version-guarded extension.

    Expiry and renewal both transition state under the same lock. Therefore a
    lease cannot be counted as expired and renewed at the same instant.
    """

    def __init__(self, on_expired: Callable[[Lease], None] | None = None, *, scan_interval_s: float = 0.01):
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._active_by_shard: dict[str, str] = {}
        self._node_active: dict[str, int] = {}
        self.on_expired = on_expired
        self.scan_interval_s = scan_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_global_leases = 0
        self.max_node_leases: dict[str, int] = {}

    def issue(
        self,
        *,
        lease_id: str,
        shard_id: str,
        node_id: str,
        config_version: int,
        generation: int,
        ttl_ms: int,
    ) -> Lease:
        if ttl_ms <= 0:
            raise BusinessError("E_LEASE", "lease ttl must be positive", field="ttl_ms", shard_id=shard_id)
        now = monotonic_ms()
        with self._lock:
            existing_id = self._active_by_shard.get(shard_id)
            if existing_id is not None:
                existing = self._leases.get(existing_id)
                if existing is not None and existing.state == "active" and now < existing.expires_ms:
                    raise BusinessError(
                        "E_LEASE",
                        "shard already has an active lease",
                        field="lease_id",
                        shard_id=shard_id,
                        detail={"active_lease_id": existing_id},
                    )
                self._transition_locked(existing, "expired")
            lease = Lease(
                lease_id=lease_id,
                shard_id=shard_id,
                node_id=node_id,
                granted_ms=now,
                expires_ms=now + ttl_ms,
                version=1,
                config_version=config_version,
                generation=generation,
                ttl_ms=ttl_ms,
            )
            self._leases[lease_id] = lease
            self._active_by_shard[shard_id] = lease_id
            self._node_active[node_id] = self._node_active.get(node_id, 0) + 1
            self.max_global_leases = max(self.max_global_leases, self._active_count_locked())
            self.max_node_leases[node_id] = max(self.max_node_leases.get(node_id, 0), self._node_active[node_id])
            return lease

    def _active_count_locked(self) -> int:
        return sum(1 for lease in self._leases.values() if lease.state == "active")

    def _transition_locked(self, lease: Lease | None, state: str) -> bool:
        if lease is None or lease.state != "active":
            return False
        lease.state = state
        if self._active_by_shard.get(lease.shard_id) == lease.lease_id:
            self._active_by_shard.pop(lease.shard_id, None)
        self._node_active[lease.node_id] = max(0, self._node_active.get(lease.node_id, 0) - 1)
        return True

    def renew(
        self,
        lease_id: str,
        expected_version: int,
        *,
        ttl_ms: int,
        node_client: NodeClient | None = None,
    ) -> Lease:
        if ttl_ms <= 0:
            raise BusinessError("E_LEASE", "lease ttl must be positive", field="ttl_ms")
        node_reply: dict[str, Any] | None = None
        with self._lock:
            lease = self._leases.get(lease_id)
            now = monotonic_ms()
            if lease is None or lease.state != "active" or now >= lease.expires_ms:
                raise BusinessError("E_LEASE", "lease absent or expired", field="lease_id")
            if lease.version != expected_version:
                raise BusinessError(
                    "E_LEASE",
                    "lease version mismatch",
                    field="lease_version",
                    detail={"expected": expected_version, "actual": lease.version},
                )
        if node_client is not None:
            node_reply = node_client.renew_lease(lease_id=lease_id, expected_version=expected_version, ttl_ms=ttl_ms)
        expired_callback: Lease | None = None
        with self._lock:
            lease = self._leases.get(lease_id)
            now = monotonic_ms()
            if lease is None or lease.state != "active":
                raise BusinessError("E_LEASE", "lease expired while renewal was in flight", field="lease_id")
            if lease.version != expected_version:
                raise BusinessError("E_LEASE", "lease version changed concurrently", field="lease_version")
            if now >= lease.expires_ms:
                self._transition_locked(lease, "expired")
                expired_callback = lease
            else:
                next_version = expected_version + 1
                if node_reply is not None and int(node_reply.get("version", -1)) != next_version:
                    raise BusinessError(
                        "E_LEASE",
                        "node returned an invalid renewed lease version",
                        field="lease_version",
                        detail={"expected": next_version, "actual": node_reply.get("version")},
                    )
                lease.version = next_version
                lease.expires_ms = now + ttl_ms
                lease.ttl_ms = ttl_ms
                if node_reply is not None and int(node_reply.get("expires_ms", lease.expires_ms)) < lease.expires_ms:
                    # Use the earlier local/node expiry so neither side overstates the lease.
                    lease.expires_ms = int(node_reply["expires_ms"])
                return lease
        if expired_callback is not None and self.on_expired is not None:
            self.on_expired(expired_callback)
        raise BusinessError("E_LEASE", "lease expired before renewal committed", field="lease_id")

    def complete(self, lease_id: str, *, expected_version: int | None = None) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if expected_version is not None and lease.version != expected_version:
                raise BusinessError("E_LEASE", "lease version mismatch at completion", field="lease_version")
            return self._transition_locked(lease, "completed")

    def cancel(self, lease_id: str) -> bool:
        with self._lock:
            return self._transition_locked(self._leases.get(lease_id), "cancelled")

    def expire_due(self, now_ms: int | None = None) -> list[Lease]:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        expired: list[Lease] = []
        with self._lock:
            for lease_id in sorted(self._leases):
                lease = self._leases[lease_id]
                if lease.state == "active" and now >= lease.expires_ms:
                    if self._transition_locked(lease, "expired"):
                        expired.append(lease)
        return expired

    def node_counts(self, node_id: str) -> int:
        with self._lock:
            return self._node_active.get(node_id, 0)

    def active_for_shard(self, shard_id: str) -> Lease | None:
        with self._lock:
            lease_id = self._active_by_shard.get(shard_id)
            if lease_id is None:
                return None
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "active" or monotonic_ms() >= lease.expires_ms:
                return None
            return lease

    def during_lease(self, lease_id: str, now_ms: int | None = None) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            now = monotonic_ms() if now_ms is None else int(now_ms)
            return bool(lease is not None and lease.state == "active" and now < lease.expires_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": [
                    self._leases[lease_id].public()
                    for lease_id in sorted(self._leases)
                    if self._leases[lease_id].state == "active"
                ],
                "node_active": {node_id: self._node_active.get(node_id, 0) for node_id in sorted(self._node_active)},
                "max_global_leases": self.max_global_leases,
                "max_node_leases": {node_id: self.max_node_leases[node_id] for node_id in sorted(self.max_node_leases)},
            }

    def start_monitor(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self.scan_interval_s):
                for lease in self.expire_due():
                    if self.on_expired is not None:
                        try:
                            self.on_expired(lease)
                        except Exception:
                            pass

        self._thread = threading.Thread(target=loop, name="lease-expiry-monitor", daemon=True)
        self._thread.start()

    def stop_monitor(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
