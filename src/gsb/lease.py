"""Batch leases: grant, optimistic renewal and real monotonic expiry."""

from __future__ import annotations

from dataclasses import dataclass

from .clock import now_ms
from .digest import short_id
from .errors import LeaseError


@dataclass(frozen=True)
class Lease:
    lease_id: str
    subscriber_id: str
    batch_id: str
    granted_at_ms: int
    expires_at_ms: int
    version: int

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "subscriber_id": self.subscriber_id,
            "batch_id": self.batch_id,
            "granted_at_ms": self.granted_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "version": self.version,
        }


class LeaseRegistry:
    """In-memory bookkeeping mirroring what is durably written to the ledger."""

    def __init__(self):
        self._leases: dict[str, Lease] = {}
        self._by_batch: dict[str, str] = {}

    def grant(self, subscriber_id: str, batch_id: str, ttl_ms: int) -> Lease:
        granted = now_ms()
        lease = Lease(
            lease_id=short_id({"lease": batch_id, "granted": granted}, 24),
            subscriber_id=subscriber_id,
            batch_id=batch_id,
            granted_at_ms=granted,
            expires_at_ms=granted + ttl_ms,
            version=1,
        )
        self._leases[lease.lease_id] = lease
        self._by_batch[batch_id] = lease.lease_id
        return lease

    def get(self, lease_id: str) -> Lease:
        return self._leases[lease_id]

    def for_batch(self, batch_id: str) -> Lease | None:
        lease_id = self._by_batch.get(batch_id)
        return self._leases.get(lease_id) if lease_id else None

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int) -> Lease:
        current = self._leases.get(lease_id)
        if current is None:
            raise LeaseError(
                f"unknown lease {lease_id}", entry=lease_id, field="lease_id"
            )
        if current.version != expected_version:
            raise LeaseError(
                f"lease version conflict: expected {current.version}, "
                f"got {expected_version}; not extended",
                entry=lease_id,
                field="version",
            )
        renewed = Lease(
            lease_id=current.lease_id,
            subscriber_id=current.subscriber_id,
            batch_id=current.batch_id,
            granted_at_ms=current.granted_at_ms,
            expires_at_ms=now_ms() + ttl_ms,
            version=current.version + 1,
        )
        self._leases[lease_id] = renewed
        return renewed

    def is_expired(self, lease: Lease, at_ms: int | None = None) -> bool:
        return (at_ms or now_ms()) >= lease.expires_at_ms

    def expired_batches(self, at_ms: int | None = None) -> list[str]:
        moment = at_ms or now_ms()
        return [
            lease.batch_id
            for lease in self._leases.values()
            if moment >= lease.expires_at_ms
        ]
