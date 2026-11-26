from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from .errors import LeaseError
from .spec import ShardSpec, Spec
from .watermark import monotonic_ms

LEASE_ACTIVE = "active"
LEASE_RELEASED = "released"
LEASE_EXPIRED = "expired"
LEASE_CANCELLED = "cancelled"
LEASE_REVOKED_ROLLBACK = "revoked_rollback"
LEASE_FAILED = "failed"
LEASE_UNKNOWN = "unknown"
TERMINAL_LEASE_STATES = {
    LEASE_RELEASED,
    LEASE_EXPIRED,
    LEASE_CANCELLED,
    LEASE_REVOKED_ROLLBACK,
    LEASE_FAILED,
    LEASE_UNKNOWN,
}


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    quota_version: int
    granted_monotonic_ms: int
    expires_monotonic_ms: int
    ttl_ms: int
    renew_count: int = 0
    version: int = 1
    state: str = LEASE_ACTIVE

    def is_expired_at(self, now_ms: int | None = None) -> bool:
        current = monotonic_ms() if now_ms is None else now_ms
        return self.state == LEASE_ACTIVE and current >= self.expires_monotonic_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseRegistry:
    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self.lock = threading.RLock()
        self.leases: dict[str, Lease] = {}
        self.by_shard: dict[str, str] = {}
        self.lease_seq = 0
        self.offline_nodes: set[str] = set()

    def mark_offline(self, node_id: str) -> None:
        with self.lock:
            self.offline_nodes.add(node_id)

    def active_leases_on_node(self, node_id: str) -> list[Lease]:
        with self.lock:
            return [lease for lease in self.leases.values() if lease.node_id == node_id and lease.state == LEASE_ACTIVE]

    def active_count_on_node(self, node_id: str) -> int:
        return len(self.active_leases_on_node(node_id))

    def active_shard_count_on_node(self, node_id: str) -> int:
        # One shard has at most one current lease; kept as a separately named
        # counter so the two declared limits are checked independently.
        return self.active_count_on_node(node_id)

    def used_resource_units(self) -> int:
        shards = self.spec.shard_map
        with self.lock:
            return sum(
                shards[lease.shard_id].resource_units
                for lease in self.leases.values()
                if lease.state == LEASE_ACTIVE and lease.shard_id in shards
            )

    def active_allocations(self) -> list[tuple[str, int, str]]:
        shards = self.spec.shard_map
        with self.lock:
            return [
                (lease.shard_id, shards[lease.shard_id].resource_units, lease.node_id)
                for lease in self.leases.values()
                if lease.state == LEASE_ACTIVE and lease.shard_id in shards
            ]

    def capacity_reasons(self, shard: ShardSpec, node_id: str) -> list[str]:
        node = self.spec.node_map[node_id]
        reasons: list[str] = []
        with self.lock:
            if node_id in self.offline_nodes:
                reasons.append("node_offline")
            used = self.used_resource_units()
            if used + shard.resource_units > self.spec.global_budget:
                reasons.append("global_budget")
            if self.active_lease_count_locked(node_id) + 1 > node.max_leases_per_node:
                reasons.append("node_lease_cap")
            if self.active_shard_count_locked(node_id) + 1 > node.concurrency_quota:
                reasons.append("node_concurrency_quota")
        return reasons

    def active_lease_count_locked(self, node_id: str) -> int:
        return sum(1 for lease in self.leases.values() if lease.node_id == node_id and lease.state == LEASE_ACTIVE)

    def active_shard_count_locked(self, node_id: str) -> int:
        return self.active_lease_count_locked(node_id)

    def admit(self, shard_id: str, node_id: str, quota_version: int, *, now_ms: int | None = None) -> Lease:
        shard = self.spec.shard_map[shard_id]
        now = monotonic_ms() if now_ms is None else now_ms
        with self.lock:
            reasons = self.capacity_reasons(shard, node_id)
            if reasons:
                raise LeaseError("lease admission rejected by capacity: " + ",".join(reasons), details={"reasons": reasons})
            if shard_id in self.by_shard:
                existing_id = self.by_shard[shard_id]
                existing = self.leases[existing_id]
                if existing.state == LEASE_ACTIVE:
                    raise LeaseError("shard already holds an active lease", object_id=shard_id)
            self.lease_seq += 1
            lease_id = f"L{self.lease_seq:06d}-{shard_id}"
            ttl = self.spec.runtime.lease_ttl_ms
            lease = Lease(
                lease_id=lease_id,
                shard_id=shard_id,
                node_id=node_id,
                quota_version=quota_version,
                granted_monotonic_ms=now,
                expires_monotonic_ms=now + ttl,
                ttl_ms=ttl,
            )
            self.leases[lease_id] = lease
            self.by_shard[shard_id] = lease_id
            return lease

    def get_shard_lease(self, shard_id: str) -> Lease | None:
        with self.lock:
            lease_id = self.by_shard.get(shard_id)
            return None if lease_id is None else self.leases[lease_id]

    def renew(
        self,
        shard_id: str,
        expected_version: int,
        *,
        node_call: Any | None = None,
        now_ms: int | None = None,
    ) -> Lease:
        now = monotonic_ms() if now_ms is None else now_ms
        with self.lock:
            lease = self.get_shard_lease(shard_id)
            if lease is None or lease.state != LEASE_ACTIVE:
                raise LeaseError("no active lease to renew", field="lease_id", object_id=shard_id)
            if lease.is_expired_at(now):
                self.transition_locked(lease, LEASE_EXPIRED)
                raise LeaseError("lease has expired and cannot be renewed", code="E_LEASE", object_id=shard_id)
            if lease.version != expected_version:
                raise LeaseError(
                    "lease version mismatch",
                    field="lease_version",
                    object_id=lease.lease_id,
                    details={"expected_version": expected_version, "current_version": lease.version},
                )
            proposed = lease.version + 1
            new_expiry = now + lease.ttl_ms
            if node_call is not None:
                response = node_call(
                    lease.node_id,
                    {
                        "type": "lease_renew",
                        "lease_id": lease.lease_id,
                        "expected_version": expected_version,
                        "proposed_version": proposed,
                        "expires_monotonic_ms": new_expiry,
                    },
                )
                if not response.get("ok"):
                    raise LeaseError("node rejected renewal", object_id=lease.lease_id, details=response)
            # Re-check after the real socket call because rollback can run concurrently.
            if lease.state != LEASE_ACTIVE or lease.version != expected_version:
                raise LeaseError(
                    "lease generation changed during node renewal",
                    object_id=lease.lease_id,
                    details={"state": lease.state, "version": lease.version},
                )
            lease.version = proposed
            lease.renew_count += 1
            lease.expires_monotonic_ms = new_expiry
            return lease

    def transition_locked(self, lease: Lease, new_state: str) -> bool:
        if lease.state != LEASE_ACTIVE:
            return False
        lease.state = new_state
        return True

    def release(self, shard_id: str, expected_version: int | None = None, *, node_call: Any | None = None) -> Lease | None:
        with self.lock:
            lease = self.get_shard_lease(shard_id)
            if lease is None:
                return None
            if expected_version is not None and lease.version != expected_version:
                raise LeaseError("lease release version mismatch", object_id=lease.lease_id)
            if node_call is not None and lease.state == LEASE_ACTIVE:
                node_call(lease.node_id, {"type": "lease_release", "lease_id": lease.lease_id})
            self.transition_locked(lease, LEASE_RELEASED)
            return lease

    def fail(self, shard_id: str, state: str = LEASE_FAILED) -> Lease | None:
        with self.lock:
            lease = self.get_shard_lease(shard_id)
            if lease is None:
                return None
            self.transition_locked(lease, state)
            return lease

    def reap_expired(self, *, now_ms: int | None = None) -> list[Lease]:
        now = monotonic_ms() if now_ms is None else now_ms
        reaped: list[Lease] = []
        with self.lock:
            for lease in list(self.leases.values()):
                if lease.state == LEASE_ACTIVE and lease.expires_monotonic_ms <= now:
                    if self.transition_locked(lease, LEASE_EXPIRED):
                        reaped.append(lease)
        return reaped

    def rollback(
        self,
        target_version: int,
        node_id: str | None = None,
        *,
        node_call: Any | None = None,
        now_ms: int | None = None,
    ) -> dict[str, list[Lease]]:
        now = monotonic_ms() if now_ms is None else now_ms
        expired: list[Lease] = []
        revoked: list[Lease] = []
        with self.lock:
            # Expiry is adjudicated first at the same timestamp. Rollback only
            # touches leases that are still unexpired, preventing double use.
            for lease in list(self.leases.values()):
                if lease.state == LEASE_ACTIVE and lease.expires_monotonic_ms <= now:
                    if self.transition_locked(lease, LEASE_EXPIRED):
                        expired.append(lease)
            for lease in list(self.leases.values()):
                if lease.state != LEASE_ACTIVE:
                    continue
                if node_id is not None and lease.node_id != node_id:
                    continue
                if lease.quota_version <= target_version:
                    continue
                if lease.expires_monotonic_ms <= now:
                    if self.transition_locked(lease, LEASE_EXPIRED):
                        expired.append(lease)
                    continue
                if self.transition_locked(lease, LEASE_REVOKED_ROLLBACK):
                    revoked.append(lease)
                    if node_call is not None:
                        try:
                            node_call(
                                lease.node_id,
                                {"type": "lease_revoke_rollback", "lease_id": lease.lease_id},
                            )
                        except Exception:
                            # The authoritative local CAS/fence still stands;
                            # restarted/dead nodes cannot advance the old token.
                            pass
        return {"expired": expired, "revoked": revoked}
