"""Lease lifecycle with real monotonic-clock expiry and optimistic renewals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock

from .clock import now_ms
from .errors import LeaseError


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_ms: int
    expire_ms: int
    renewals: int
    lease_version: int
    capacity_version: int
    state: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Lease":
        return cls(**payload)


@dataclass
class Decision:
    adjudication: int
    at_ms: int
    action: str
    lease_id: str
    shard_id: str
    node_id: str
    capacity_version: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class LeaseManager:
    """Owns the authoritative lease view for one orchestration process."""

    def __init__(self, ttl_ms: int):
        self.ttl_ms = ttl_ms
        self._lock = Lock()
        self._leases: dict[str, Lease] = {}
        self._shard_owner: dict[str, str] = {}
        self._counter = 0
        self.decisions: list[Decision] = []

    def _next_adjudication(self) -> int:
        self._counter += 1
        return self._counter

    def _record(self, action: str, lease: Lease, reason: str) -> Decision:
        decision = Decision(
            adjudication=self._next_adjudication(),
            at_ms=now_ms(),
            action=action,
            lease_id=lease.lease_id,
            shard_id=lease.shard_id,
            node_id=lease.node_id,
            capacity_version=lease.capacity_version,
            reason=reason,
        )
        self.decisions.append(decision)
        return decision

    def grant(
        self,
        lease_id: str,
        shard_id: str,
        node_id: str,
        capacity_version: int,
    ) -> Lease:
        with self._lock:
            if shard_id in self._shard_owner:
                existing = self._leases[self._shard_owner[shard_id]]
                raise LeaseError(
                    f"shard {shard_id} already has active lease {existing.lease_id}",
                    path=f"leases.{shard_id}",
                )
            granted = now_ms()
            lease = Lease(
                lease_id=lease_id,
                shard_id=shard_id,
                node_id=node_id,
                granted_ms=granted,
                expire_ms=granted + self.ttl_ms,
                renewals=0,
                lease_version=1,
                capacity_version=capacity_version,
                state="active",
            )
            self._leases[lease_id] = lease
            self._shard_owner[shard_id] = lease_id
            self._record("grant", lease, "scheduled")
            return lease

    def get(self, lease_id: str) -> Lease | None:
        with self._lock:
            lease = self._leases.get(lease_id)
            return None if lease is None else Lease(**asdict(lease))

    def owner_of(self, shard_id: str) -> Lease | None:
        with self._lock:
            lease_id = self._shard_owner.get(shard_id)
            if lease_id is None:
                return None
            return Lease(**asdict(self._leases[lease_id]))

    def active_leases(self) -> list[Lease]:
        with self._lock:
            self._expire_locked(now_ms())
            return [
                Lease(**asdict(lease))
                for lease in self._leases.values()
                if lease.state == "active"
            ]

    def active_for_node(self, node_id: str) -> list[Lease]:
        return [lease for lease in self.active_leases() if lease.node_id == node_id]

    def is_expired(self, lease_id: str, at_ms: int | None = None) -> bool:
        moment = now_ms() if at_ms is None else at_ms
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "active":
                return True
            return moment >= lease.expire_ms

    def _expire_locked(self, moment: int) -> list[Lease]:
        expired: list[Lease] = []
        for lease in list(self._leases.values()):
            if lease.state == "active" and moment >= lease.expire_ms:
                lease.state = "expired"
                self._shard_owner.pop(lease.shard_id, None)
                self._record("expire", lease, "monotonic_deadline")
                expired.append(lease)
        return expired

    def expire_due(self) -> list[Lease]:
        with self._lock:
            return self._expire_locked(now_ms())

    def release(self, lease_id: str, reason: str = "completed") -> Lease | None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return None
            if lease.state == "active":
                lease.state = "released"
                self._shard_owner.pop(lease.shard_id, None)
                self._record("release", lease, reason)
            return Lease(**asdict(lease))

    def apply_renewal_receipt(self, lease_id: str, receipt: dict) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError(f"unknown lease {lease_id}", path=f"leases.{lease_id}")
            if not receipt.get("accepted"):
                raise LeaseError(
                    f"node rejected renewal for {lease_id}: {receipt.get('error', 'version_mismatch')}",
                    path=f"leases.{lease_id}.lease_version",
                )
            if receipt["lease_version"] != lease.lease_version + 1:
                raise LeaseError(
                    "renewal receipt version is not the expected successor",
                    path=f"leases.{lease_id}.lease_version",
                )
            lease.renewals += 1
            lease.lease_version = receipt["lease_version"]
            lease.expire_ms = receipt["expire_ms"]
            self._record("renew", lease, "optimistic_check_passed")
            return Lease(**asdict(lease))

    def reject_renewal(self, lease_id: str, reason: str) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is not None:
                self._record("renew_reject", lease, reason)

    def revoke_before_version(self, capacity_version: int) -> list[Lease]:
        with self._lock:
            revoked: list[Lease] = []
            for lease in list(self._leases.values()):
                if lease.state == "active" and lease.capacity_version < capacity_version:
                    lease.state = "revoked"
                    self._shard_owner.pop(lease.shard_id, None)
                    self._record("revoke", lease, f"capacity_replaced_v{capacity_version}")
                    revoked.append(Lease(**asdict(lease)))
            return revoked

    def decisions_snapshot(self) -> list[dict]:
        with self._lock:
            return [decision.to_dict() for decision in self.decisions]
