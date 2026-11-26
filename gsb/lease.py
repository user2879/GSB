from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .clock import monotonic_ms
from .errors import LeaseError


ACTIVE_STATUSES = {"granted", "renewed"}
TERMINAL_STATUSES = {"completed", "expired", "revoked", "reclaimed", "failed"}


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    capacity_version: int
    resource_units: int
    granted_monotonic_ms: int
    expires_monotonic_ms: int
    lease_version: int = 1
    status: str = "granted"
    released: bool = False
    release_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseManager:
    def __init__(self, *, ttl_ms: int = 5000) -> None:
        self.ttl_ms = int(ttl_ms)
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._by_shard: dict[str, str] = {}

    def grant(
        self,
        *,
        shard_id: str,
        node_id: str,
        capacity_version: int,
        resource_units: int,
        ttl_ms: int | None = None,
        now_ms: int | None = None,
        lease_id: str | None = None,
    ) -> Lease:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        ttl = self.ttl_ms if ttl_ms is None else int(ttl_ms)
        lease = Lease(
            lease_id=lease_id or f"lease-{uuid.uuid4().hex}",
            shard_id=shard_id,
            node_id=node_id,
            capacity_version=int(capacity_version),
            resource_units=int(resource_units),
            granted_monotonic_ms=now,
            expires_monotonic_ms=now + ttl,
        )
        with self._lock:
            existing_id = self._by_shard.get(shard_id)
            if existing_id and self._leases[existing_id].active:
                raise LeaseError(
                    f"shard {shard_id} already has active lease {existing_id}",
                    field="shard_id",
                    position=shard_id,
                )
            self._leases[lease.lease_id] = lease
            self._by_shard[shard_id] = lease.lease_id
            return lease

    def get(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError(f"unknown lease {lease_id}", field="lease_id", position=lease_id)
            return lease

    def for_shard(self, shard_id: str) -> Lease | None:
        with self._lock:
            lease_id = self._by_shard.get(shard_id)
            return self._leases.get(lease_id) if lease_id else None

    def renew_locally(
        self,
        lease_id: str,
        expected_version: int,
        *,
        now_ms: int | None = None,
        ttl_ms: int | None = None,
        new_version: int | None = None,
    ) -> Lease:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        with self._lock:
            lease = self.get(lease_id)
            if lease.lease_version != int(expected_version):
                raise LeaseError(
                    "lease optimistic version mismatch; expiry was not extended",
                    field="lease_version",
                    position=lease_id,
                    expected_version=lease.lease_version,
                    presented_version=expected_version,
                )
            if not lease.active:
                raise LeaseError(
                    f"cannot renew terminal lease in status {lease.status}",
                    field="status",
                    position=lease_id,
                )
            if now >= lease.expires_monotonic_ms:
                lease.status = "expired"
                raise LeaseError(
                    "lease has already expired",
                    field="expires_monotonic_ms",
                    position=lease_id,
                )
            ttl = self.ttl_ms if ttl_ms is None else int(ttl_ms)
            lease.expires_monotonic_ms = now + ttl
            lease.lease_version = (
                int(new_version) if new_version is not None else lease.lease_version + 1
            )
            lease.status = "renewed"
            return lease

    def expire_due(self, *, now_ms: int | None = None) -> list[Lease]:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        expired: list[Lease] = []
        with self._lock:
            for lease in self._leases.values():
                if lease.active and now >= lease.expires_monotonic_ms:
                    lease.status = "expired"
                    expired.append(lease)
        return expired

    def revoke_old_versions(
        self,
        current_versions: dict[str, int] | int,
        *,
        global_version: bool = False,
    ) -> list[Lease]:
        revoked: list[Lease] = []
        with self._lock:
            for lease in self._leases.values():
                if not lease.active:
                    continue
                if global_version:
                    current = int(current_versions)  # type: ignore[arg-type]
                    stale = lease.capacity_version < current
                else:
                    versions = current_versions  # type: ignore[assignment]
                    current = int(versions.get(lease.node_id, lease.capacity_version))  # type: ignore[union-attr]
                    stale = lease.capacity_version < current
                if stale:
                    lease.status = "revoked"
                    revoked.append(lease)
        return revoked

    def transition(self, lease_id: str, status: str) -> Lease:
        with self._lock:
            lease = self.get(lease_id)
            if not lease.active and status in ACTIVE_STATUSES:
                raise LeaseError(f"cannot reactivate {lease.status} lease", position=lease_id)
            lease.status = status
            return lease

    def release_once(
        self,
        lease_id: str,
        reason: str,
        *,
        status: str | None = None,
    ) -> Lease | None:
        """Atomically mark capacity released exactly once.

        Returns the lease only for the caller that wins the release transition;
        duplicate expiry/revocation/recovery calls receive None and therefore
        must not subtract budget a second time.
        """

        with self._lock:
            lease = self.get(lease_id)
            if lease.released:
                return None
            lease.released = True
            lease.release_reason = reason
            if status is not None:
                lease.status = status
            elif lease.active:
                lease.status = reason if reason in TERMINAL_STATUSES else "reclaimed"
            return lease

    def active_for_node(self, node_id: str) -> list[Lease]:
        with self._lock:
            return [lease for lease in self._leases.values() if lease.node_id == node_id and lease.active]

    def active_count_for_node(self, node_id: str) -> int:
        return len(self.active_for_node(node_id))

    def all_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.to_dict() for lease in self._leases.values()]
