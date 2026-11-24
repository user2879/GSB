"""Sliding-window lease lifecycle.

Leases guard each window aggregation. Renewal uses optimistic concurrency via
the lease version: a stale version is rejected with ``E_LEASE`` and never
extends the lease. Expiry is decided from a monotonic clock (real timing),
never from wall-clock time or artificial sleeps.
"""
from __future__ import annotations

import itertools
import threading
import uuid
from dataclasses import replace
from typing import Optional

from .clock import MonotonicClock, SystemClock
from .errors import LeaseError
from .models import Lease


class LeaseManager:
    def __init__(self, clock: MonotonicClock | None = None,
                 ttl_ms: int = 60_000):
        self._clock = clock or SystemClock()
        self._ttl_ms = int(ttl_ms)
        self._lock = threading.RLock()
        # key -> lease
        self._leases: dict[tuple[str, int], Lease] = {}
        # lease_id -> key (identity index)
        self._index: dict[str, tuple[str, int]] = {}
        self._counter = itertools.count(1)

    @property
    def ttl_ms(self) -> int:
        return self._ttl_ms

    def _key(self, metric_key: str, window_index: int) -> tuple[str, int]:
        return metric_key, window_index

    def grant(self, metric_key: str, window_index: int, start_seq: int,
              end_seq: int) -> Lease:
        """Grant a fresh lease; an existing active lease must be released."""
        with self._lock:
            key = self._key(metric_key, window_index)
            existing = self._leases.get(key)
            now = self._clock.now_ms()
            if existing is not None and existing.state == "active" \
                    and not existing.is_expired(now):
                raise LeaseError(
                    f"active lease {existing.lease_id} already guards "
                    f"{metric_key} window {window_index}",
                    field="lease_id",
                    detail={"metric_key": metric_key,
                            "window_index": window_index})
            lease_id = f"lease-{next(self._counter)}-{uuid.uuid4().hex[:12]}"
            lease = Lease(
                lease_id=lease_id,
                metric_key=metric_key,
                window_index=window_index,
                start_seq=start_seq,
                end_seq=end_seq,
                granted_ms=now,
                expiry_ms=now + self._ttl_ms,
                version=1,
                state="active",
            )
            self._leases[key] = lease
            self._index[lease_id] = key
            return lease

    def renew(self, lease_id: str, expected_version: int,
              ttl_ms: Optional[int] = None) -> Lease:
        """Renew with optimistic version check."""
        with self._lock:
            key = self._index.get(lease_id)
            if key is None:
                raise LeaseError(f"unknown lease id {lease_id!r}",
                                 field="lease_id")
            lease = self._leases[key]
            if lease.version != expected_version:
                raise LeaseError(
                    f"lease version conflict: expected {expected_version}, "
                    f"current {lease.version}; renewal rejected",
                    field="version",
                    detail={"lease_id": lease_id,
                            "expected_version": expected_version,
                            "current_version": lease.version})
            if lease.state != "active":
                raise LeaseError(
                    f"lease {lease_id} is {lease.state} and cannot renew",
                    field="state")
            now = self._clock.now_ms()
            if lease.is_expired(now):
                self._leases[key] = replace(lease, state="expired")
                raise LeaseError(
                    f"lease {lease_id} expired at {lease.expiry_ms} "
                    f"(now {now}); renewal rejected",
                    field="expiry_ms",
                    detail={"expiry_ms": lease.expiry_ms, "now_ms": now})
            ttl = self._ttl_ms if ttl_ms is None else int(ttl_ms)
            renewed = Lease(
                lease_id=lease.lease_id,
                metric_key=lease.metric_key,
                window_index=lease.window_index,
                start_seq=lease.start_seq,
                end_seq=lease.end_seq,
                granted_ms=lease.granted_ms,
                expiry_ms=now + ttl,
                version=lease.version + 1,
                state="active",
            )
            self._leases[key] = renewed
            return renewed

    def release(self, lease_id: str) -> None:
        with self._lock:
            key = self._index.get(lease_id)
            if key is None:
                return
            lease = self._leases[key]
            self._leases[key] = replace(lease, state="revoked")

    def reap_expired(self) -> list[Lease]:
        """Mark all genuinely expired leases expired using the clock."""
        with self._lock:
            now = self._clock.now_ms()
            expired: list[Lease] = []
            for key, lease in list(self._leases.items()):
                if lease.state == "active" and lease.is_expired(now):
                    marked = replace(lease, state="expired")
                    self._leases[key] = marked
                    expired.append(marked)
            return expired

    def active_lease(self, metric_key: str, window_index: int) -> Optional[Lease]:
        with self._lock:
            lease = self._leases.get(self._key(metric_key, window_index))
            if lease is None:
                return None
            if lease.state == "active" and not lease.is_expired(
                    self._clock.now_ms()):
                return lease
            return None

    def is_active(self, metric_key: str, window_index: int) -> bool:
        return self.active_lease(metric_key, window_index) is not None

    def all_leases(self) -> list[Lease]:
        with self._lock:
            self.reap_expired()
            return [self._leases[k] for k in sorted(self._leases)]

    def restore(self, lease: Lease) -> None:
        """Rebuild in-memory state from a committed ledger snapshot."""
        with self._lock:
            key = self._key(lease.metric_key, lease.window_index)
            self._leases[key] = lease
            self._index[lease.lease_id] = key

    def clear(self) -> None:
        with self._lock:
            self._leases.clear()
            self._index.clear()
