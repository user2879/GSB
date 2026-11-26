"""Real monotonic-clock leases with one-shot release semantics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .errors import EdgeError, LeaseError


@dataclass
class Lease:
    lease_id: str
    node_id: str
    version: int
    granted_monotonic_ms: int
    expires_monotonic_ms: int
    state: str = "active"
    release_reason: str | None = None

    def is_expired(self, now_ms: int | None = None) -> bool:
        now = now_ms if now_ms is not None else int(time.monotonic() * 1000)
        return now >= self.expires_monotonic_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "version": self.version,
            "granted_monotonic_ms": self.granted_monotonic_ms,
            "expires_monotonic_ms": self.expires_monotonic_ms,
            "state": self.state,
            "release_reason": self.release_reason,
        }


class LeaseCapacityTimeout(EdgeError):
    def __init__(self, node_id: str) -> None:
        super().__init__(
            "E_LEASE",
            f"lease capacity unavailable for node {node_id}",
            details={"node_id": node_id},
            exit_code=5,
        )


class LeaseManager:
    def __init__(self, lease_limit: int, ttl_ms: int):
        self._lease_limit = int(lease_limit)
        self._ttl_ms = int(ttl_ms)
        self._condition = threading.Condition()
        self._active: dict[str, Lease] = {}
        self._node_active: dict[str, int] = {}
        self._version_counter: dict[str, int] = {}
        self._sequence = 0
        self._one_shot_release: dict[str, str] = {}
        self._expired: list[Lease] = []

    def _expire_locked(self, now_ms: int) -> list[Lease]:
        expired: list[Lease] = []
        for lease in sorted(self._active.values(), key=lambda item: item.lease_id):
            if lease.is_expired(now_ms):
                lease.state = "expired"
                lease.release_reason = "expired"
                self._one_shot_release[lease.lease_id] = "expired"
                self._node_active[lease.node_id] -= 1
                del self._active[lease.lease_id]
                expired.append(lease)
        if expired:
            self._expired.extend(expired)
            self._condition.notify_all()
        return expired

    def acquire(
        self,
        node_id: str,
        node_limit: int,
        ttl_ms: int | None = None,
        timeout_s: float = 10.0,
    ) -> Lease:
        ttl = self._ttl_ms if ttl_ms is None else int(ttl_ms)
        if ttl <= 0:
            raise ValueError("ttl_ms must be positive")
        with self._condition:
            def capacity_available() -> bool:
                self._expire_locked(int(time.monotonic() * 1000))
                return self._node_active.get(node_id, 0) < min(node_limit, self._lease_limit)

            if not self._condition.wait_for(capacity_available, timeout=timeout_s):
                raise LeaseCapacityTimeout(node_id)
            now_ms = int(time.monotonic() * 1000)
            self._sequence += 1
            version = self._version_counter.get(node_id, 0) + 1
            self._version_counter[node_id] = version
            lease = Lease(
                lease_id=f"lease-{node_id}-{self._sequence}",
                node_id=node_id,
                version=version,
                granted_monotonic_ms=now_ms,
                expires_monotonic_ms=now_ms + ttl,
            )
            self._active[lease.lease_id] = lease
            self._node_active[node_id] = self._node_active.get(node_id, 0) + 1
            return lease

    def release(self, lease: Lease, reason: str = "completed") -> bool:
        with self._condition:
            if lease.state != "active" or lease.lease_id not in self._active:
                return False
            lease.state = reason
            lease.release_reason = reason
            self._one_shot_release[lease.lease_id] = reason
            self._node_active[lease.node_id] -= 1
            del self._active[lease.lease_id]
            self._condition.notify_all()
            return True

    def expire_due(self) -> list[Lease]:
        with self._condition:
            return self._expire_locked(int(time.monotonic() * 1000))

    def update_after_renewal(self, lease: Lease, expected_version: int, new_expiry_ms: int) -> None:
        with self._condition:
            current = self._active.get(lease.lease_id)
            if current is not lease or current.state != "active":
                raise LeaseError(lease.lease_id, expected_version, None)
            if current.version != expected_version:
                raise LeaseError(lease.lease_id, expected_version, current.version)
            current.version = expected_version + 1
            current.expires_monotonic_ms = int(new_expiry_ms)

    def note_renewal_rejection(self, lease: Lease, expected_version: int, actual_version: int | None) -> None:
        if expected_version != actual_version:
            raise LeaseError(lease.lease_id, expected_version, actual_version)

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "active": [
                    lease.to_dict()
                    for lease in sorted(self._active.values(), key=lambda item: item.lease_id)
                ],
                "expired": [lease.to_dict() for lease in self._expired],
                "one_shot_release": dict(sorted(self._one_shot_release.items())),
                "version_counter": dict(sorted(self._version_counter.items())),
            }

