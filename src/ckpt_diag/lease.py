"""Requirement 4: shard leases with real monotonic-clock expiry.

A lease is held by at most one batch at a time. Renewal is an optimistic
concurrency check keyed on the lease version: a stale version is rejected with
E_LEASE and the expiry is never extended.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import count
from typing import Callable

from .errors import LeaseError

Clock = Callable[[], int]


@dataclass(frozen=True)
class Lease:
    lease_id: str
    shard_id: str
    granted_ms: int
    expires_ms: int
    renew_count: int
    lease_version: int
    batch_id: str | None

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_ms


class LeaseManager:
    """In-memory lease authority used by both parent and shard process."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._seq = count(1)
        self._leases: dict[str, Lease] = {}
        self._batch_holder: dict[str, str] = {}  # batch_id -> lease_id

    def issue(self, shard_id: str, ttl_ms: int, batch_id: str) -> Lease:
        now_ms = self._clock()
        if batch_id in self._batch_holder:
            raise LeaseError(f"batch {batch_id} already holds a lease",
                             f"shard:{shard_id}", "batch_id")
        lease_id = f"lease-{next(self._seq):08d}"
        lease = Lease(lease_id, shard_id, now_ms, now_ms + max(ttl_ms, 0),
                      renew_count=0, lease_version=1, batch_id=batch_id)
        self._leases[lease_id] = lease
        self._batch_holder[batch_id] = lease_id
        return lease

    def get(self, lease_id: str) -> Lease:
        if lease_id not in self._leases:
            raise LeaseError(f"unknown lease {lease_id}", "lease", "lease_id")
        return self._leases[lease_id]

    def is_active(self, lease_id: str) -> bool:
        lease = self.get(lease_id)
        return not lease.is_expired(self._clock())

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int) -> Lease:
        lease = self.get(lease_id)
        if lease.lease_version != expected_version:
            raise LeaseError(
                f"optimistic lease version mismatch: expected {expected_version}, "
                f"actual {lease.lease_version}; expiry unchanged",
                f"lease:{lease_id}", "lease_version",
                detail={"expires_ms": lease.expires_ms})
        if lease.is_expired(self._clock()):
            raise LeaseError(f"lease {lease_id} already expired and cannot renew",
                             f"lease:{lease_id}", "expires_ms")
        now_ms = self._clock()
        renewed = replace(lease, expires_ms=now_ms + max(ttl_ms, 0),
                          renew_count=lease.renew_count + 1,
                          lease_version=lease.lease_version + 1)
        self._leases[lease_id] = renewed
        return renewed

    def release(self, lease_id: str, expected_version: int | None = None) -> Lease:
        lease = self.get(lease_id)
        if expected_version is not None and lease.lease_version != expected_version:
            raise LeaseError(
                f"release version mismatch: expected {expected_version}, "
                f"actual {lease.lease_version}",
                f"lease:{lease_id}", "lease_version")
        if lease.batch_id is not None:
            self._batch_holder.pop(lease.batch_id, None)
        return lease

    def expire_due(self) -> list[Lease]:
        """Return leases whose monotonic deadline has passed (real expiry)."""
        now_ms = self._clock()
        due = [lease for lease in self._leases.values()
               if not lease.is_expired(now_ms - 1) and lease.is_expired(now_ms)]
        return due

    def active_leases_for_shard(self, shard_id: str) -> list[Lease]:
        now_ms = self._clock()
        return [lease for lease in self._leases.values()
                if lease.shard_id == shard_id and not lease.is_expired(now_ms)]
