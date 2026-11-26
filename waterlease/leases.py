from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

from .errors import LeaseConflictError
from .process_manager import ProcessManager
from .timeutil import monotonic_ns


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    capacity_version: str
    granted_at_ns: int
    expires_at_ns: int
    ttl_ms: int
    renew_count: int = 0
    version: int = 1
    status: str = "active"

    def is_expired(self, now_ns: int | None = None) -> bool:
        current = time.monotonic_ns() if now_ns is None else now_ns
        return self.status == "active" and current >= self.expires_at_ns

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["granted_at_monotonic_ms"] = self.granted_at_ns // 1_000_000
        data["expires_at_monotonic_ms"] = self.expires_at_ns // 1_000_000
        data["remaining_ttl_ms"] = max(0, (self.expires_at_ns - time.monotonic_ns()) // 1_000_000)
        data.pop("granted_at_ns", None)
        data.pop("expires_at_ns", None)
        return data


class LeaseManager:
    def __init__(self, processes: ProcessManager, ttl_ms: int) -> None:
        self.processes = processes
        self.ttl_ms = ttl_ms
        self._leases: dict[str, Lease] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        *,
        shard_id: str,
        node_id: str,
        capacity_version: str,
        lease_id: str | None = None,
    ) -> Lease:
        lease_id = lease_id or f"lease-{shard_id}-{uuid.uuid4().hex[:12]}"
        now = monotonic_ns()
        lease = Lease(
            lease_id=lease_id,
            shard_id=shard_id,
            node_id=node_id,
            capacity_version=capacity_version,
            granted_at_ns=now,
            expires_at_ns=now + self.ttl_ms * 1_000_000,
            ttl_ms=self.ttl_ms,
        )
        handle = self.processes.handles[node_id]
        handle.client_message(
            "lease_register",
            lease_id=lease_id,
            shard_id=shard_id,
            capacity_version=capacity_version,
        )
        with self._lock:
            self._leases[lease_id] = lease
        return lease

    def get(self, lease_id: str) -> Lease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def renew(self, lease_id: str, expected_version: int | None = None) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseConflictError("unknown lease", path="$.lease_id", details={"lease_id": lease_id})
            if lease.status != "active":
                raise LeaseConflictError(
                    f"lease is {lease.status}; renewal rejected",
                    path="$.status",
                    details={"lease_id": lease_id, "status": lease.status},
                )
            if lease.is_expired():
                lease.status = "expired"
                raise LeaseConflictError(
                    "lease has expired according to CLOCK_MONOTONIC; not extended",
                    path="$.expires_at_monotonic_ms",
                    details={"lease_id": lease_id, "version": lease.version},
                )
            expected = lease.version if expected_version is None else expected_version
            if expected != lease.version:
                raise LeaseConflictError(
                    "local optimistic lease version mismatch; expiry was not extended",
                    path="$.version",
                    details={
                        "lease_id": lease_id,
                        "expected_version": expected,
                        "actual_version": lease.version,
                    },
                )
            handle = self.processes.handles[lease.node_id]
            response = handle.client_message(
                "lease_renew",
                lease_id=lease_id,
                shard_id=lease.shard_id,
                capacity_version=lease.capacity_version,
                expected_version=expected,
            )
            lease.version = int(response.get("lease_version", lease.version + 1))
            lease.renew_count = int(response.get("renew_count", lease.renew_count + 1))
            lease.expires_at_ns = monotonic_ns() + self.ttl_ms * 1_000_000
            return lease

    def release(self, lease_id: str, expected_version: int | None = None) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseConflictError("unknown lease", path="$.lease_id")
            expected = lease.version if expected_version is None else expected_version
            try:
                handle = self.processes.handles[lease.node_id]
                handle.client_message(
                    "lease_release",
                    lease_id=lease_id,
                    shard_id=lease.shard_id,
                    capacity_version=lease.capacity_version,
                    expected_version=expected,
                )
            except LeaseConflictError:
                pass
            lease.status = "released"
            return lease

    def expire_due(self) -> list[Lease]:
        expired: list[Lease] = []
        with self._lock:
            for lease in self._leases.values():
                if lease.is_expired():
                    lease.status = "expired"
                    expired.append(lease)
                    try:
                        handle = self.processes.handles.get(lease.node_id)
                        if handle is not None:
                            handle.client_message(
                                "lease_release",
                                lease_id=lease.lease_id,
                                shard_id=lease.shard_id,
                                capacity_version=lease.capacity_version,
                                expected_version=lease.version,
                            )
                    except Exception:
                        pass
        return expired

    def active_lease_count(self, node_id: str) -> int:
        with self._lock:
            return sum(
                1
                for lease in self._leases.values()
                if lease.node_id == node_id and lease.status == "active" and not lease.is_expired()
            )

    def active_global_count(self) -> int:
        with self._lock:
            return sum(1 for lease in self._leases.values() if lease.status == "active")
