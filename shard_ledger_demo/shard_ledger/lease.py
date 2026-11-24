"""Shard leases: grant, optimistic renew and real monotonic-clock expiry."""

from dataclasses import dataclass

from .clock import now_ms
from .errors import LeaseError


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_ms: int
    expires_ms: int
    version: int
    units: int
    active: bool = True

    def to_dict(self):
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "version": self.version,
            "units": self.units,
            "active": self.active,
        }


class LeaseManager:
    def __init__(self, ttl_ms: int, on_expire=None):
        self.ttl_ms = ttl_ms
        self.on_expire = on_expire
        self._leases = {}
        self._node_active = {}
        self._global_units = 0
        self._counter = 0

    def can_grant(self, node_id: str, units: int, node_lease_limit: int, global_budget: int) -> bool:
        return (
            self._node_active.get(node_id, 0) < node_lease_limit
            and self._global_units + units <= global_budget
        )

    def grant(self, shard_id: str, node_id: str, units: int) -> Lease:
        self._counter += 1
        current = now_ms()
        lease = Lease(
            lease_id=f"L{self._counter:06d}",
            shard_id=shard_id,
            node_id=node_id,
            granted_ms=current,
            expires_ms=current + self.ttl_ms,
            version=1,
            units=units,
        )
        self._leases[lease.lease_id] = lease
        self._node_active[node_id] = self._node_active.get(node_id, 0) + 1
        self._global_units += units
        return lease

    def renew(self, lease_id: str, expected_version: int) -> Lease:
        lease = self._leases[lease_id]
        if not lease.active or now_ms() >= lease.expires_ms:
            raise LeaseError("lease is expired and cannot be renewed", entity=lease.shard_id, field="lease")
        if lease.version != expected_version:
            raise LeaseError(
                "optimistic lease version mismatch; original lease is unchanged",
                entity=lease.shard_id,
                field="lease_version",
                expected=expected_version,
                actual=lease.version,
            )
        lease.version += 1
        lease.expires_ms = now_ms() + self.ttl_ms
        return lease

    def release(self, lease_id: str):
        lease = self._leases.get(lease_id)
        if lease is None or not lease.active:
            return
        lease.active = False
        self._node_active[lease.node_id] -= 1
        self._global_units -= lease.units

    def sweep_expired(self):
        expired = []
        current = now_ms()
        for lease_id in sorted(self._leases):
            lease = self._leases[lease_id]
            if lease.active and current >= lease.expires_ms:
                self.release(lease_id)
                expired.append(lease)
                if self.on_expire is not None:
                    self.on_expire(lease)
        return expired

    def active_count(self, node_id: str) -> int:
        return self._node_active.get(node_id, 0)

    def global_units(self) -> int:
        return self._global_units

    def snapshot(self) -> list:
        return [self._leases[key].to_dict() for key in sorted(self._leases)]
