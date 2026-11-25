"""Lease issuance, optimistic revocation, real expiry and unique release.

Three resource-release paths (freeze, expiry, revoke) converge on a single
idempotent arbitration point per ``(node, shard)``.  Whichever path wins
owns the release once; later paths observe ``released=True`` and cannot
return the resource units a second time.
"""

from dataclasses import dataclass, asdict
from threading import Lock
from typing import Dict, List, Optional

from .clock import Clock
from .errors import LeaseError


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    grant_ms: int
    expiry_ms: int
    version: int
    demand: int
    released: bool = False
    release_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class LeaseManager:
    def __init__(self, clock: Clock):
        self._clock = clock
        self._lock = Lock()
        self._by_key: Dict[str, Lease] = {}       # (node, shard) -> latest lease
        self._by_id: Dict[str, Lease] = {}
        self._active: Dict[str, List[Lease]] = {}  # node -> active leases
        self._versions: Dict[str, int] = {}       # (node, shard) -> monotonic version
        self._budget_used = 0
        self._release_log: List[dict] = []

    @staticmethod
    def _key(node_id, shard_id):
        return f"{node_id}|{shard_id}"

    def grant(self, node_id, shard_id, demand, ttl_ms) -> Lease:
        key = self._key(node_id, shard_id)
        now = self._clock.now_ms()
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None and not existing.released:
                raise LeaseError(
                    f"active lease already held for {shard_id} on {node_id}",
                    object_id=shard_id, field="lease_id",
                )
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            lease = Lease(
                lease_id=f"L{version:04d}-{node_id}-{shard_id}",
                shard_id=shard_id,
                node_id=node_id,
                grant_ms=now,
                expiry_ms=now + ttl_ms,
                version=version,
                demand=demand,
            )
            self._by_key[key] = lease
            self._by_id[lease.lease_id] = lease
            self._active.setdefault(node_id, []).append(lease)
            self._budget_used += demand
            return lease

    def get(self, node_id, shard_id) -> Optional[Lease]:
        return self._by_key.get(self._key(node_id, shard_id))

    def active_lease(self, node_id, shard_id) -> Optional[Lease]:
        lease = self.get(node_id, shard_id)
        if lease is None or lease.released:
            return None
        return lease

    def node_active(self, node_id) -> List[Lease]:
        return [lease for lease in self._active.get(node_id, []) if not lease.released]

    def node_active_count(self, node_id) -> int:
        return len(self.node_active(node_id))

    def node_inflight_count(self, node_id) -> int:
        return self.node_active_count(node_id)

    def budget_used(self) -> int:
        return self._budget_used

    def expire_due(self) -> List[dict]:
        """Release every active lease past its monotonic deadline."""
        now = self._clock.now_ms()
        results = []
        with self._lock:
            for lease in list(self._by_id.values()):
                if not lease.released and lease.expiry_ms <= now:
                    results.append(self._release_locked(lease, "expiry"))
        return [r for r in results if r is not None]

    def revoke(self, node_id, shard_id, expected_version) -> dict:
        """Optimistic revocation; wrong/missing version raises E_LEASE."""
        lease = self.active_lease(node_id, shard_id)
        if lease is None:
            raise LeaseError(
                f"no active lease for {shard_id} on {node_id}",
                object_id=shard_id, field="version",
            )
        if lease.version != expected_version:
            raise LeaseError(
                f"lease version conflict: expected {expected_version}, "
                f"current {lease.version}",
                object_id=shard_id, field="version",
                detail={"expected": expected_version, "current": lease.version},
            )
        with self._lock:
            return self._release_locked(lease, "revoke")

    def release_for_freeze(self, node_id, shard_id) -> Optional[dict]:
        """Release path used when a shard is pushed back to the frozen edge."""
        lease = self.active_lease(node_id, shard_id)
        if lease is None:
            return None
        with self._lock:
            return self._release_locked(lease, "freeze")

    def release_on_complete(self, node_id, shard_id) -> Optional[dict]:
        """Normal terminal release; same arbitration point as contested paths."""
        lease = self.active_lease(node_id, shard_id)
        if lease is None:
            return None
        with self._lock:
            return self._release_locked(lease, "complete")

    def _release_locked(self, lease: Lease, reason: str) -> Optional[dict]:
        if lease.released:
            # Unique ownership: the first path won; never double-return budget.
            return None
        lease.released = True
        lease.release_reason = reason
        self._active[lease.node_id] = [
            item for item in self._active.get(lease.node_id, []) if not item.released
        ]
        self._budget_used -= lease.demand
        event = {
            "lease_id": lease.lease_id,
            "shard_id": lease.shard_id,
            "node_id": lease.node_id,
            "version": lease.version,
            "demand": lease.demand,
            "reason": reason,
            "at_ms": self._clock.now_ms(),
        }
        self._release_log.append(event)
        return event

    def release_log(self) -> List[dict]:
        return list(self._release_log)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "leases": [lease.to_dict() for lease in self._by_id.values()],
                "versions": dict(self._versions),
                "budget_used": self._budget_used,
                "release_log": list(self._release_log),
            }

    def restore_snapshot(self, snapshot: dict, active_only: bool):
        """Rebuild manager state from committed ledger records.

        ``active_only`` keeps only leases whose release was never committed;
        those represent in-flight work recoverable after a hard kill.
        """
        with self._lock:
            self._by_key.clear()
            self._by_id.clear()
            self._active.clear()
            self._versions.update(snapshot.get("versions", {}))
            self._release_log = list(snapshot.get("release_log", []))
            self._budget_used = 0
            for row in snapshot.get("leases", []):
                lease = Lease(**row)
                if active_only and lease.released:
                    continue
                self._by_id[lease.lease_id] = lease
                self._by_key[self._key(lease.node_id, lease.shard_id)] = lease
                if not lease.released:
                    self._active.setdefault(lease.node_id, []).append(lease)
                    self._budget_used += lease.demand
