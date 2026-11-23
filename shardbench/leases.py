"""Concurrency leases with real monotonic-clock expiry.

A lease carries: lease id, target physical db, grant time (ms), expiry time
(ms) and a version number. Renewal is optimistic on the version: a stale
version is rejected with E_LEASE and the lease is NOT extended. Expiry is
judged against the local monotonic clock; expired leases are reaped and the
db's concurrency allowance is released with a recorded reclaim reason.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from .clock import monotonic_ms
from .errors import BizError, E_LEASE


@dataclass
class Lease:
    lease_id: str
    db: int
    granted_ms: int
    expires_ms: int
    version: int = 1
    state: str = "held"  # held | released | expired | unknown
    reclaim_reason: str = ""

    def as_dict(self):
        return {
            "lease_id": self.lease_id,
            "db": self.db,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "version": self.version,
            "state": self.state,
            "reclaim_reason": self.reclaim_reason,
        }


class LeaseManager:
    def __init__(self, ttl_ms: int, clock=monotonic_ms, listener=None):
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be positive")
        self.ttl_ms = ttl_ms
        self.clock = clock
        self.listener = listener  # callable(event, lease_dict)
        self._leases = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def _emit(self, event, lease):
        if self.listener:
            self.listener(event, lease.as_dict())

    def acquire(self, db: int) -> Lease:
        with self._lock:
            self._reap_locked()
            now = self.clock()
            lease = Lease(
                lease_id=f"lease-{next(self._counter)}",
                db=db,
                granted_ms=now,
                expires_ms=now + self.ttl_ms,
            )
            self._leases[lease.lease_id] = lease
        self._emit("lease_acquired", lease)
        return lease

    def renew(self, lease_id: str, expected_version: int, extend_ms: int = None) -> Lease:
        with self._lock:
            self._reap_locked()
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "held":
                raise BizError(
                    E_LEASE,
                    f"lease {lease_id} is not held",
                    field="lease_id",
                    detail={"lease_id": lease_id},
                )
            if lease.version != expected_version:
                raise BizError(
                    E_LEASE,
                    f"lease version conflict: expected {expected_version}, "
                    f"current {lease.version}",
                    field="version",
                    detail={
                        "lease_id": lease_id,
                        "expected_version": expected_version,
                        "current_version": lease.version,
                    },
                )
            lease.version += 1
            lease.expires_ms = self.clock() + (extend_ms or self.ttl_ms)
        self._emit("lease_renewed", lease)
        return lease

    def release(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise BizError(E_LEASE, f"unknown lease {lease_id}", field="lease_id")
            if lease.state == "held":
                lease.state = "released"
        self._emit("lease_released", lease)
        return lease

    def get(self, lease_id: str) -> Lease:
        with self._lock:
            self._reap_locked()
            return self._leases.get(lease_id)

    def active_count(self, db: int = None) -> int:
        with self._lock:
            self._reap_locked()
            return sum(
                1
                for l in self._leases.values()
                if l.state == "held" and (db is None or l.db == db)
            )

    def held_leases(self):
        with self._lock:
            self._reap_locked()
            return [l for l in self._leases.values() if l.state == "held"]

    def reap(self):
        with self._lock:
            return self._reap_locked()

    def _reap_locked(self):
        now = self.clock()
        reclaimed = []
        for lease in self._leases.values():
            if lease.state == "held" and now >= lease.expires_ms:
                lease.state = "expired"
                lease.reclaim_reason = (
                    f"ttl_expired: now={now}ms >= expires={lease.expires_ms}ms; "
                    f"db {lease.db} concurrency allowance released"
                )
                reclaimed.append(lease)
        for lease in reclaimed:
            self._emit("lease_expired", lease)
        return reclaimed
