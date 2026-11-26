"""Lease issuance, optimistic renewal and expiry arbitration.

A lease turns a worker concurrency slot into a stateful resource with a real
monotonic-clock deadline.  Expiry and batch cancellation can both reclaim a
slot; :class:`LeaseManager` guarantees that each reclaim is attributed to
exactly one cause so the slot is never released twice or double-counted.
"""

import itertools
import threading

from .digest import short_digest
from .errors import LeaseError


class MonotonicClock:
    """Millisecond monotonic clock; overridable for deterministic tests."""

    def now_ms(self):
        import time

        return time.monotonic_ns() // 1_000_000


class Lease:
    def __init__(self, lease_id, worker_id, batch_id, slot_index,
                 issued_at, expiry_ms, version=1):
        self.lease_id = lease_id
        self.worker_id = worker_id
        self.batch_id = batch_id
        self.slot_index = slot_index
        self.issued_at = issued_at
        self.expiry_ms = expiry_ms
        self.version = version
        self.released_by = None
        self.history = []

    @property
    def active(self):
        return self.released_by is None

    def snapshot(self):
        return {
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "batch_id": self.batch_id,
            "slot_index": self.slot_index,
            "issued_at": self.issued_at,
            "expiry_ms": self.expiry_ms,
            "version": self.version,
            "released_by": self.released_by,
            "history": list(self.history),
        }


class LeaseManager:
    def __init__(self, spec, clock=None):
        self.spec = spec
        self.clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._leases = {}
        self._slots = {
            worker.worker_id: set(range(worker.slot_cap))
            for worker in spec.workers
        }

    def active_count(self):
        with self._lock:
            return sum(1 for lease in self._leases.values() if lease.active)

    def active_for_worker(self, worker_id):
        with self._lock:
            return sum(
                1 for lease in self._leases.values()
                if lease.active and lease.worker_id == worker_id
            )

    def can_grant(self, worker_id):
        cap = self.spec.lease_policy.global_max_active
        if self.active_count() >= cap:
            return False
        if self.active_for_worker(worker_id) >= self.spec.worker_by_id[
            worker_id
        ].slot_cap:
            return False
        return True

    def grant(self, worker_id, batch_id):
        """Grant a lease or raise E_LEASE when no slot is available."""
        with self._lock:
            now = self.clock.now_ms()
            cap = self.spec.lease_policy.global_max_active
            if self.active_count() >= cap:
                raise LeaseError(
                    "global active lease ceiling reached",
                    field="global_max_active",
                )
            worker = self.spec.worker_by_id[worker_id]
            if self.active_for_worker(worker_id) >= worker.slot_cap:
                raise LeaseError(
                    f"worker {worker_id!r} slot capacity reached",
                    field="slot_cap",
                )
            free = self._slots[worker_id]
            if not free:
                raise LeaseError(
                    f"worker {worker_id!r} has no free slot",
                    field="slot_cap",
                )
            slot_index = min(free)
            free.remove(slot_index)
            duration = self.spec.lease_policy.lease_duration_ms
            lease_id = "L" + short_digest(
                ["lease", worker_id, batch_id, next(self._ids), now]
            )
            lease = Lease(
                lease_id, worker_id, batch_id, slot_index, now,
                now + duration,
            )
            lease.history.append(
                {"event": "grant", "at": now, "expiry_ms": lease.expiry_ms}
            )
            self._leases[lease_id] = lease
            return lease

    def renew(self, lease_id, expected_version, extension_ms=None):
        """Optimistic renewal; a stale version raises E_LEASE, no extension."""
        with self._lock:
            lease = self._require_active(lease_id)
            now = self.clock.now_ms()
            if lease.version != expected_version:
                raise LeaseError(
                    f"lease version conflict: expected {lease.version}, "
                    f"got {expected_version}; expiry unchanged",
                    field="version",
                )
            if extension_ms is None:
                extension_ms = self.spec.lease_policy.lease_duration_ms
            lease.version += 1
            lease.expiry_ms = max(lease.expiry_ms, now) + extension_ms
            lease.history.append(
                {"event": "renew", "at": now, "version": lease.version,
                 "expiry_ms": lease.expiry_ms}
            )
            return lease

    def sweep_expired(self):
        """Reclaim every lease past its deadline; returns reclaimed ids."""
        reclaimed = []
        with self._lock:
            now = self.clock.now_ms()
            for lease in self._leases.values():
                if lease.active and lease.expiry_ms <= now:
                    self._reclaim(lease, "expiry", now)
                    reclaimed.append(lease.lease_id)
        return sorted(reclaimed)

    def cancel(self, lease_id):
        """Reclaim a slot due to batch cancellation.

        Idempotent: if expiry already reclaimed the slot the second reclaim is
        refused (returns ``False``) so the slot has exactly one owner of the
        release decision.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError(f"unknown lease {lease_id!r}",
                                  field="lease_id")
            if not lease.active:
                return False
            self._reclaim(lease, "cancel", self.clock.now_ms())
            return True

    def _reclaim(self, lease, cause, now):
        lease.released_by = cause
        lease.history.append(
            {"event": "release", "cause": cause, "at": now}
        )
        self._slots[lease.worker_id].add(lease.slot_index)

    def _require_active(self, lease_id):
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError(f"unknown lease {lease_id!r}",
                              field="lease_id")
        if not lease.active:
            raise LeaseError(
                f"lease {lease_id!r} already released by "
                f"{lease.released_by!r}",
                field="lease_id",
            )
        return lease

    def get(self, lease_id):
        with self._lock:
            return self._leases[lease_id]

    def all_snapshots(self):
        with self._lock:
            return [
                lease.snapshot()
                for lease in sorted(
                    self._leases.values(), key=lambda item: item.lease_id
                )
            ]
