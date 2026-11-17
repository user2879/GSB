"""Lease state machine with monotonic-clock expiry and real cache reclamation.

States: granted -> active -> (expired | revoked). ``unknown`` is a terminal
state assigned during recovery to leases that were not settled before a
forced kill. Every transition records reason, sequence progress and the
monotonic timestamp.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .errors import E_BUSY, E_SCOPE, E_SEQ, PlatformError

GRANTED = "granted"
ACTIVE = "active"
EXPIRED = "expired"
REVOKED = "revoked"
UNKNOWN = "unknown"
TERMINAL = {EXPIRED, REVOKED, UNKNOWN}


@dataclass
class Lease:
    lease_id: str
    node_id: str
    sectors: tuple
    seq_lo: int
    seq_hi: int
    grant_mono: float
    expire_mono: float
    state: str = GRANTED
    progress: int = 0

    def to_dict(self):
        return {
            "lease_id": self.lease_id, "node_id": self.node_id,
            "sectors": list(self.sectors), "seq_lo": self.seq_lo,
            "seq_hi": self.seq_hi, "grant_mono": self.grant_mono,
            "expire_mono": self.expire_mono, "state": self.state,
            "progress": self.progress,
        }

    @staticmethod
    def from_dict(d):
        return Lease(d["lease_id"], d["node_id"], tuple(d["sectors"]),
                     d["seq_lo"], d["seq_hi"], d["grant_mono"],
                     d["expire_mono"], d["state"], d.get("progress", 0))


class LeaseManager:
    def __init__(self, store=None, clock=time.monotonic):
        self.store = store
        self.clock = clock
        self._leases = {}
        self._cache = {}
        self._lock = threading.RLock()
        self.transitions = []

    def _transition(self, lease, to, reason):
        frm = lease.state
        lease.state = to
        mono = self.clock()
        rec = {"lease_id": lease.lease_id, "from": frm, "to": to,
               "reason": reason, "progress": lease.progress, "mono": mono}
        self.transitions.append(rec)
        if to in TERMINAL:
            self._cache.pop(lease.lease_id, None)
        if self.store:
            self.store.log_transition(lease.lease_id, frm, to, reason,
                                      lease.progress, mono)
            self.store.update_lease(lease.lease_id, to, lease.progress)
        return rec

    def grant(self, lease_id, node_id, sectors, seq_lo, seq_hi, ttl_ms):
        with self._lock:
            self.reap()
            for sid in sectors:
                if not self.sector_free(sid):
                    raise PlatformError(E_BUSY, sector=sid,
                                        reason="sector covered by non-terminal lease")
            now = self.clock()
            lease = Lease(lease_id, node_id, tuple(sorted(sectors)), seq_lo,
                          seq_hi, now, now + ttl_ms / 1000.0, state="none")
            self._leases[lease_id] = lease
            self._cache[lease_id] = lease
            self._transition(lease, GRANTED, "granted")
            if self.store:
                self.store.insert_lease(lease)
            return lease

    def activate(self, lease_id):
        with self._lock:
            lease = self._require(lease_id)
            if lease.state == GRANTED:
                self._transition(lease, ACTIVE, "node_ack")
            return lease

    def complete(self, lease_id, progress):
        with self._lock:
            lease = self._require(lease_id)
            lease.progress = progress
            self._transition(lease, REVOKED, "completed")
            return lease

    def fail(self, lease_id, reason):
        with self._lock:
            lease = self._require(lease_id)
            self._transition(lease, REVOKED, reason)
            return lease

    def mark_unknown(self, lease_id, reason="restart_unsettled"):
        with self._lock:
            lease = self._require(lease_id)
            self._transition(lease, UNKNOWN, reason)
            return lease

    def reap(self):
        now = self.clock()
        expired = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.state not in TERMINAL and lease.expire_mono <= now:
                    self._transition(lease, EXPIRED, "ttl_expired")
                    expired.append(lease.lease_id)
        return expired

    def sector_free(self, sector):
        return all(sector not in l.sectors or l.state in TERMINAL
                   for l in self._leases.values())

    def get(self, lease_id):
        return self._leases.get(lease_id)

    def cached(self, lease_id):
        return self._cache.get(lease_id)

    def all_leases(self):
        return list(self._leases.values())

    def check_access(self, lease_id, sectors, seq):
        """Validate an access against the lease; raises E_SCOPE / E_SEQ."""
        with self._lock:
            self.reap()
            lease = self._cache.get(lease_id)
            if lease is None:
                raise PlatformError(E_SCOPE, lease=lease_id,
                                    reason="lease expired or recycled; cache miss")
            outside = sorted(set(sectors) - set(lease.sectors))
            if outside:
                raise PlatformError(E_SCOPE, lease=lease_id, sector=outside[0],
                                    reason="sector outside lease scope")
            if not (lease.seq_lo <= seq <= lease.seq_hi):
                raise PlatformError(E_SEQ, lease=lease_id, seq=seq,
                                    seq_lo=lease.seq_lo, seq_hi=lease.seq_hi,
                                    reason="sequence outside lease range")

    def replace_all(self, lease_dicts):
        with self._lock:
            self._leases.clear()
            self._cache.clear()
            for d in lease_dicts:
                lease = Lease.from_dict(d)
                self._leases[lease.lease_id] = lease
                if lease.state not in TERMINAL:
                    self._cache[lease.lease_id] = lease

    def load(self, lease_dicts):
        with self._lock:
            for d in lease_dicts:
                lease = Lease.from_dict(d)
                self._leases[lease.lease_id] = lease
                if lease.state not in TERMINAL:
                    self._cache[lease.lease_id] = lease

    def _require(self, lease_id):
        lease = self._leases.get(lease_id)
        if lease is None:
            raise PlatformError(E_SCOPE, lease=lease_id, reason="unknown lease")
        return lease
