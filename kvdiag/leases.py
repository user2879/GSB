"""Lease issuing, optimistic-version renewal and real-clock expiry."""

import itertools
import threading

from .clock import mono_ms
from .errors import LeaseError

LEASE_LIVE = "live"
LEASE_EXPIRED = "expired"


class LeaseManager:
    """Issues and renews leases for index positions of each shard.

    Renewal uses the lease version for optimistic concurrency; a stale
    version is rejected with E_LEASE. Expiry is judged against the local
    monotonic clock. Per-shard in-flight renewals are bounded by the
    declared quota; excess requests queue and their wait start/end
    monotonic milliseconds are recorded.
    """

    def __init__(self, spec, logger=None):
        self.spec = spec
        self.logger = logger
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._leases = {}            # lease_id -> lease dict
        self._by_shard = {s.shard_id: [] for s in spec.shards}
        self._inflight = {s.shard_id: 0 for s in spec.shards}
        self._expiry_pos = {s.shard_id: 0 for s in spec.shards}
        self.wait_events = []        # {shard_id, task_kind, wait_start, wait_end}

    def issue(self, shard_id, key_digest, batch_id=0):
        shard = self.spec.shard(shard_id)
        now = mono_ms()
        lease_id = "lease-%04d-%06d" % (batch_id, next(self._seq))
        lease = {
            "lease_id": lease_id,
            "shard_id": shard_id,
            "key_digest": key_digest,
            "version": 1,
            "issued_mono_ms": now,
            "expiry_mono_ms": now + shard.lease.ttl_ms,
            "state": LEASE_LIVE,
            "batch_id": batch_id,
        }
        with self._lock:
            self._leases[lease_id] = lease
            self._by_shard[shard_id].append(lease_id)
        return dict(lease)

    def _get(self, lease_id):
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("unknown lease", details={"lease_id": lease_id})
        return lease

    def renew(self, lease_id, expected_version):
        """Optimistic renewal; version mismatch is rejected with E_LEASE."""
        with self._lock:
            lease = self._get(lease_id)
            if lease["state"] != LEASE_LIVE:
                raise LeaseError("lease is not live", shard_id=lease["shard_id"],
                                 details={"lease_id": lease_id,
                                          "state": lease["state"]})
            if expected_version != lease["version"]:
                raise LeaseError(
                    "lease version mismatch",
                    shard_id=lease["shard_id"],
                    details={"lease_id": lease_id,
                             "expected_version": expected_version,
                             "actual_version": lease["version"]})
            shard = self.spec.shard(lease["shard_id"])
            lease["version"] += 1
            lease["expiry_mono_ms"] = mono_ms() + shard.lease.ttl_ms
            return dict(lease)

    def acquire_renewal_slot(self, shard_id, task_kind="renew", timeout_ms=None):
        """Block until a per-shard in-flight renewal slot is available.

        Records the queue wait window in monotonic milliseconds.
        """
        shard = self.spec.shard(shard_id)
        quota = shard.lease.max_inflight_renewals
        start = mono_ms()
        deadline = None if timeout_ms is None else start + timeout_ms
        while True:
            with self._lock:
                if self._inflight[shard_id] < quota:
                    self._inflight[shard_id] += 1
                    end = mono_ms()
                    event = {"shard_id": shard_id, "task_kind": task_kind,
                             "wait_start_mono_ms": start, "wait_end_mono_ms": end}
                    self.wait_events.append(event)
                    if self.logger:
                        self.logger.emit("renewal_slot_acquired", shard_id=shard_id,
                                         wait_start_mono_ms=start,
                                         wait_end_mono_ms=end)
                    return event
            if deadline is not None and mono_ms() >= deadline:
                raise LeaseError("renewal slot wait timed out", shard_id=shard_id,
                                 details={"quota": quota})
            threading.Event().wait(0.001)

    def release_renewal_slot(self, shard_id):
        with self._lock:
            self._inflight[shard_id] = max(0, self._inflight[shard_id] - 1)

    def expire_due(self, now_ms=None):
        """Expire every live lease whose deadline passed (real monotonic clock).

        Reclamation releases the index position occupied by the lease and
        advances the shard expiry position to its stable value.
        """
        now = mono_ms() if now_ms is None else now_ms
        expired = []
        with self._lock:
            for lease in self._leases.values():
                if lease["state"] == LEASE_LIVE and lease["expiry_mono_ms"] <= now:
                    lease["state"] = LEASE_EXPIRED
                    expired.append(dict(lease))
        for lease in sorted(expired, key=lambda l: (l["shard_id"], l["lease_id"])):
            with self._lock:
                self._expiry_pos[lease["shard_id"]] += 1
            if self.logger:
                self.logger.emit("lease_expired", shard_id=lease["shard_id"],
                                 lease_id=lease["lease_id"],
                                 expiry_pos=self._expiry_pos[lease["shard_id"]])
        return expired

    def expiry_position(self, shard_id):
        with self._lock:
            return self._expiry_pos[shard_id]

    def live_leases(self, shard_id):
        with self._lock:
            return [dict(self._leases[lid]) for lid in self._by_shard[shard_id]
                    if self._leases[lid]["state"] == LEASE_LIVE]

    def all_leases(self):
        with self._lock:
            return [dict(l) for l in self._leases.values()]

    def all_leases_for(self, shard_id):
        with self._lock:
            return [dict(self._leases[lid]) for lid in self._by_shard[shard_id]]
