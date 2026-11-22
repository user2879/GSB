"""Per-domain lease arbitration state machine.

The state machine is single-threaded by construction: each domain arbiter
processes one request at a time inside its own subprocess, so no internal lock
is needed. Expiry is evaluated against the arbiter's own monotonic clock on
*every* request and on explicit :meth:`reap` calls driven by the service loop;
the demo therefore relies on real timing rather than artificial delays.

Permission table (state transitions)
------------------------------------
    (none)      -> granted   ACQUIRE with free capacity
    (none)      -> pending   ACQUIRE with full capacity and queue room
    pending     -> granted   capacity freed, stable queue drain
    pending     -> (removed) RELEASE while waiting
    granted     -> renewed   RENEW strictly before expiry
    renewed     -> renewed   further RENEW strictly before expiry
    granted/renewed -> expired   monotonic now >= expiry (reaper only)
    granted/renewed -> released  RELEASE before expiry
    *           -> rejected/verdict with reason otherwise
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..clock import Clock, MonotonicClock
from ..models import (
    Domain,
    Receipt,
    ScaleFactor,
    Transition,
    Watermark,
)


@dataclass
class _Waiter:
    seq: int
    holder_id: str
    priority: int
    enqueued_clock_ms: int

    def sort_key(self) -> tuple[int, int]:
        # priority DESC, seq ASC -> negate priority for ascending sort.
        return (-self.priority, self.seq)


@dataclass
class _Lease:
    holder_id: str
    state: str  # granted | renewed
    granted_seq: int
    started_ms: int
    expires_ms: int


class DomainArbiter:
    """Deterministic arbitration engine for one domain."""

    def __init__(self, domain: Domain, scale: ScaleFactor,
                 clock: Clock | None = None):
        self.domain = domain
        self.scale = scale
        self.clock: Clock = clock or MonotonicClock()
        self._leases: "OrderedDict[str, _Lease]" = OrderedDict()
        self._waiters: list[_Waiter] = []
        self._handled_seqs: set[int] = set()
        self._last_receipt_by_seq: dict[int, Receipt] = {}
        self.transitions: list[Transition] = []
        self._last_reap_transitions: list[Transition] = []

    # ------------------------------------------------------------------ views
    def snapshot(self) -> dict[str, Any]:
        now = self.clock.now_ms()
        active = [
            {"holder_id": l.holder_id, "state": l.state,
             "remaining_ms": max(0, l.expires_ms - now),
             "started_ms": l.started_ms, "expires_ms": l.expires_ms,
             "granted_seq": l.granted_seq}
            for l in self._leases.values()
        ]
        queue = [{"seq": w.seq, "holder_id": w.holder_id, "priority": w.priority}
                 for w in self._waiters]
        return {
            "domain_id": self.domain.domain_id,
            "domain_version": self.domain.version,
            "capacity": self.domain.capacity,
            "active_leases": active,
            "queue": queue,
            "watermark": self._watermark(seq=0).__dict__,
        }

    def active_count(self) -> int:
        return len(self._leases)

    # ----------------------------------------------------------------- expiry
    def reap(self, seq: int | None = None) -> list[Transition]:
        """Expire every lease whose deadline has passed (real monotonic time).

        Returns the expiry transitions and immediately drains the queue for any
        freed capacity. Renewal and expiry for the same instant cannot both win
        because request handling always reaps first and expiry is ``now >= end``.
        """
        now = self.clock.now_ms()
        expired: list[str] = []
        for holder_id, lease in list(self._leases.items()):
            if now >= lease.expires_ms:
                expired.append(holder_id)
        made: list[Transition] = []
        for holder_id in expired:
            lease = self._leases.pop(holder_id)
            made.append(self._record(
                seq=seq if seq is not None else lease.granted_seq,
                holder_id=holder_id, state_from=lease.state, state_to="expired",
                reason="lease_deadline_reached", started=lease.started_ms,
                expires=lease.expires_ms, clock_ms=now))
        if expired:
            made.extend(self._drain(seq=seq, clock_ms=now))
        self._last_reap_transitions = made
        return made

    # ---------------------------------------------------------------- requests
    def handle(self, seq: int, holder_id: str, action: str,
               priority: int) -> Receipt:
        # Idempotent re-delivery: duplicate seq returns the committed receipt.
        if seq in self._handled_seqs:
            return self._duplicate(seq)

        # Real-time expiry sweep precedes every arbitration decision.
        reap_made = self.reap(seq=seq)
        now = self.clock.now_ms()

        if action == "ACQUIRE":
            return self._acquire(seq, holder_id, priority, now)
        if action == "RENEW":
            return self._renew(seq, holder_id, now)
        if action == "RELEASE":
            receipt = self._release(seq, holder_id, now)
            receipt.drain_transitions = list(receipt.drain_transitions or []) + reap_made
            return receipt
        receipt = None
        if action == "ACQUIRE":
            receipt = self._acquire(seq, holder_id, priority, now)
        elif action == "RENEW":
            receipt = self._renew(seq, holder_id, now)
        if receipt is not None and reap_made:
            receipt.drain_transitions = list(receipt.drain_transitions or []) + reap_made
        if receipt is not None:
            return receipt
        raise ValueError(f"unknown action {action!r}")

    def _acquire(self, seq: int, holder_id: str, priority: int, now: int) -> Receipt:
        # Re-acquisition by a holder that still holds a lease is idempotent.
        if holder_id in self._leases:
            lease = self._leases[holder_id]
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "GRANTED", lease, now,
                                 state=lease.state, reason="already_held")

        if len(self._leases) < self.domain.capacity:
            lease = self._grant(seq, holder_id, now, reason="capacity_available")
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "GRANTED", lease, now,
                                 state=lease.state, reason="capacity_available")

        if len(self._waiters) >= self.domain.queue_limit:
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "REJECTED_QUEUE_FULL", None, now,
                                 state="pending", reason="queue_limit_reached")

        waiter = _Waiter(seq, holder_id, priority, now)
        self._waiters.append(waiter)
        # Stable ordering, never relies on dict/insertion iteration for grants.
        self._waiters.sort(key=lambda w: w.sort_key())
        self._record(seq, holder_id, None, "pending", "capacity_full_queued",
                     started=None, expires=None, clock_ms=now)
        self._handled_seqs.add(seq)
        return self._receipt(seq, holder_id, "QUEUED", None, now,
                             state="pending", reason="capacity_full_queued")

    def _renew(self, seq: int, holder_id: str, now: int) -> Receipt:
        lease = self._leases.get(holder_id)
        if lease is None:
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "RENEW_NOT_HELD", None, now,
                                 state="released", reason="no_active_lease")
        # reap() already ran, so a surviving lease is strictly unexpired here.
        if now >= lease.expires_ms:  # defensive, normally unreachable
            self._leases.pop(holder_id)
            self._record(seq, holder_id, lease.state, "expired",
                         "renewal_at_or_after_expiry", lease.started_ms,
                         lease.expires_ms, now)
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "RENEW_EXPIRED", None, now,
                                 state="expired", reason="renewal_at_or_after_expiry")

        prev_state = lease.state
        lease.state = "renewed"
        lease.started_ms = now
        lease.expires_ms = now + self.domain.lease_duration_ms
        self._record(seq, holder_id, prev_state, "renewed",
                     "renewal_before_expiry", lease.started_ms, lease.expires_ms, now)
        self._handled_seqs.add(seq)
        return self._receipt(seq, holder_id, "RENEWED", lease, now,
                             state="renewed", reason="renewal_before_expiry")

    def _release(self, seq: int, holder_id: str, now: int) -> Receipt:
        # A waiting holder can cancel its pending request.
        for idx, waiter in enumerate(self._waiters):
            if waiter.holder_id == holder_id:
                removed = self._waiters.pop(idx)
                self._record(seq, holder_id, "pending", "released",
                             "waiting_request_cancelled", None, None, now)
                self._handled_seqs.add(seq)
                return self._receipt(seq, holder_id, "RELEASED", None, now,
                                     state="released",
                                     reason="waiting_request_cancelled")

        lease = self._leases.pop(holder_id, None)
        if lease is None:
            self._handled_seqs.add(seq)
            return self._receipt(seq, holder_id, "RELEASE_NOT_HELD", None, now,
                                 state="released", reason="no_active_lease")

        self._record(seq, holder_id, lease.state, "released",
                     "explicit_release", lease.started_ms, lease.expires_ms, now)
        drain_made = self._drain(seq=seq, clock_ms=now)
        self._handled_seqs.add(seq)
        now2 = self.clock.now_ms()
        receipt = self._receipt(seq, holder_id, "RELEASED", None, now2,
                                 state="released", reason="explicit_release")
        receipt.drain_transitions = drain_made
        return receipt

    # ------------------------------------------------------------- queue drain
    def _drain(self, seq: int | None, clock_ms: int) -> list[Transition]:
        made: list[Transition] = []
        while self._waiters and len(self._leases) < self.domain.capacity:
            waiter = self._waiters.pop(0)  # list is kept in stable sorted order
            # _grant inserts the lease into the active table first; the two
            # explicit migrations below make the pending -> granted movement
            # durable so ledger replay reconstructs waiter removal and the lease.
            lease = self._grant(waiter.seq, waiter.holder_id, clock_ms,
                                reason="queue_drain_after_capacity_freed")
            made.append(self._record(
                waiter.seq, waiter.holder_id, "pending", "granted",
                "queue_drain_pending_exit", None, None, clock_ms))
            made.append(self._record(
                waiter.seq, waiter.holder_id, "pending", "granted",
                "queue_drain_after_capacity_freed", lease.started_ms,
                lease.expires_ms, clock_ms))
        return made

    def _grant(self, seq: int, holder_id: str, now: int, reason: str) -> _Lease:
        lease = _Lease(
            holder_id=holder_id, state="granted", granted_seq=seq,
            started_ms=now, expires_ms=now + self.domain.lease_duration_ms)
        self._leases[holder_id] = lease
        self._record(seq, holder_id, None, "granted", reason,
                     lease.started_ms, lease.expires_ms, now)
        return lease

    # ---------------------------------------------------------------- helpers
    def _record(self, seq: int, holder_id: str, state_from: str | None,
                state_to: str, reason: str, started: int | None,
                expires: int | None, clock_ms: int) -> Transition:
        tr = Transition(
            seq=seq, domain_id=self.domain.domain_id, holder_id=holder_id,
            state_from=state_from, state_to=state_to, reason=reason,
            started_ms=started, expires_ms=expires,
            domain_version=self.domain.version, clock_ms=clock_ms)
        self.transitions.append(tr)
        return tr

    def _watermark(self, seq: int) -> Watermark:
        active = len(self._leases)
        cap = self.domain.capacity
        return Watermark(
            domain_id=self.domain.domain_id, seq=seq, active_leases=active,
            capacity=cap, free_capacity=cap - active,
            queue_length=len(self._waiters),
            ratio_fixed=self.scale.ratio_fixed(active, cap),
            scale=self.scale.scale, truncation="floor",
            error_bound_fixed=self.scale.error_bound_fixed())

    def _receipt(self, seq: int, holder_id: str, verdict: str,
                 lease: _Lease | None, now: int, state: str,
                 reason: str) -> Receipt:
        remaining = max(0, lease.expires_ms - now) if lease else 0
        r = Receipt(
            seq=seq, domain_id=self.domain.domain_id, holder_id=holder_id,
            verdict=verdict, remaining_ms=remaining, clock_ms=now, state=state,
            queue_length=len(self._waiters), active_leases=len(self._leases),
            free_capacity=self.domain.capacity - len(self._leases),
            ratio_fixed=self.scale.ratio_fixed(len(self._leases), self.domain.capacity),
            scale=self.scale.scale,
            error_bound_fixed=self.scale.error_bound_fixed(), reason=reason)
        self._last_receipt_by_seq[seq] = r
        return r

    def _duplicate(self, seq: int) -> Receipt:
        r = self._last_receipt_by_seq[seq]
        dup = Receipt(**{**r.as_dict(), "verdict": "DUPLICATE",
                         "reason": "duplicate_event_redelivery"})
        return dup
