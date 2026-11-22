"""Deterministic lease adjudication engine.

The engine is pure logic parameterised by an integer event-time clock: the
runtime supplies real monotonic milliseconds and replay supplies the persisted
event timestamps, so the same request sequence always converges to the same
ownership and watermark conclusions.
"""
from __future__ import annotations

from .errors import E_LEASE, LeaseError
from .model import (
    ACTIVE_STATES, Decision, Event, Lease, Pool, Request,
    EV_DENY, EV_EXPIRE, EV_GRANT, EV_RENEW, EV_REVOKE, EV_ROLLBACK,
    GRANTED, RENEWED, next_state, transition_allowed,
)


class Engine:
    def __init__(self, pools: dict[str, Pool], rollback_targets: dict) -> None:
        self.pools = pools
        self.rollback_targets = dict(rollback_targets)
        self.leases: dict[str, Lease] = {}
        # Active ownership index: (pool_id, key) -> lease_id.
        self.active_index: dict[tuple[str, str], str] = {}
        # Where each key is currently active (at most one lease at any time).
        self.key_location: dict[str, str] = {}
        self.events: list[Event] = []
        self.decisions: dict[str, Decision] = {}
        self._seq = 0
        self._lease_serial: dict[tuple[str, str], int] = {}
        self._post_dead_denied: set[str] = set()

    # ------------------------------------------------------------------ utils
    def _emit(self, event_type: str, at_ms: int, *, request_id=None, key=None,
              pool_id=None, lease_id=None, revision=None, priority=None,
              detail=None) -> Event:
        self._seq += 1
        event = Event(self._seq, event_type, request_id, key, pool_id,
                      lease_id, at_ms, revision, priority, detail or {})
        self.events.append(event)
        return event

    def _new_lease_id(self, pool_id: str, key: str) -> str:
        token = (pool_id, key)
        serial = self._lease_serial.get(token, 0) + 1
        self._lease_serial[token] = serial
        return f"L-{key}-{pool_id}-{serial:03d}"

    def _active_leases(self, pool_id: str) -> list[Lease]:
        return [lease for lease in self.leases.values()
                if lease.pool_id == pool_id and lease.active]

    def _latest_lease_for(self, pool_id: str, key: str) -> Lease | None:
        matches = [lease for lease in self.leases.values()
                   if lease.pool_id == pool_id and lease.key == key]
        if not matches:
            return None
        return max(matches, key=lambda lease: lease.grant_ms)

    def _used(self, pool_id: str) -> int:
        return len(self._active_leases(pool_id))

    def _move(self, lease: Lease, action: str) -> str:
        if not transition_allowed(lease.state, action):
            raise KeyError(
                f"illegal transition {lease.state} --{action}--> for "
                f"{lease.lease_id}")
        lease.state = next_state(lease.state, action)
        return lease.state

    def _release(self, lease: Lease) -> None:
        if self.active_index.get((lease.pool_id, lease.key)) == lease.lease_id:
            del self.active_index[(lease.pool_id, lease.key)]
        if self.key_location.get(lease.key) == lease.pool_id:
            del self.key_location[lease.key]

    # ------------------------------------------------------------- lifecycle
    def tick(self, now_ms: int) -> list[Event]:
        """Expire every active lease whose deadline has passed.

        Leases are scanned in lease_id order so expiry outcomes are stable;
        timing itself comes from the injected monotonic clock.
        """
        produced: list[Event] = []
        for lease in sorted(self.leases.values(), key=lambda item: item.lease_id):
            if lease.active and lease.expire_ms <= now_ms:
                state = self._move(lease, "EXPIRE")
                assert state == "expired"
                self._release(lease)
                produced.append(self._emit(
                    EV_EXPIRE, lease.expire_ms, key=lease.key,
                    pool_id=lease.pool_id, lease_id=lease.lease_id,
                    revision=lease.revision, detail={"renew_count":
                                                     lease.renew_count}))
        return produced

    def submit(self, request: Request, now_ms: int) -> Decision:
        """Adjudicate one request at monotonic event-time ``now_ms``."""
        if request.request_id in self.decisions:
            return self.decisions[request.request_id]
        self.tick(now_ms)
        pool = self.pools[request.pool_id]
        existing_id = self.active_index.get((request.pool_id, request.key))

        if existing_id is not None:
            return self._renew(request, now_ms, self.leases[existing_id])
        # A prior lease that is no longer active cannot be renewed. One
        # explicit E_LEASE is raised against the dead lease; after that a fresh
        # request may grant a replacement lease (capacity was released).
        prior = self._latest_lease_for(request.pool_id, request.key)
        if prior is not None and not prior.active and prior.lease_id \
                not in self._post_dead_denied:
            self._post_dead_denied.add(prior.lease_id)
            return self._deny(request, now_ms, prior.lease_id,
                              f"previous lease is {prior.state}")
        if self._used(request.pool_id) < pool.capacity:
            return self._grant(request, now_ms)
        return self._preempt_or_deny(request, now_ms, pool)

    # ---------------------------------------------------------------- renew
    def _renew(self, request: Request, now_ms: int, lease: Lease) -> Decision:
        if not lease.active:
            return self._deny(request, now_ms, lease.lease_id,
                              "lease is no longer active")
        if lease.expire_ms <= now_ms:
            return self._deny(request, now_ms, lease.lease_id,
                              "renewal arrived at or after expiry")
        if request.priority != lease.priority:
            return self._deny(request, now_ms, lease.lease_id,
                              "only the current holder may renew (priority)")
        self._move(lease, "RENEW")
        lease.renew_count += 1
        lease.last_renew_ms = now_ms
        lease.revoke_boundary_ms = now_ms
        lease.expire_ms = now_ms + self.pools[lease.pool_id].lease_ms
        lease.revision = request.revision
        self._emit(EV_RENEW, now_ms, request_id=request.request_id,
                   key=lease.key, pool_id=lease.pool_id,
                   lease_id=lease.lease_id, revision=lease.revision,
                   priority=lease.priority,
                   detail={"renew_count": lease.renew_count,
                           "expire_ms": lease.expire_ms})
        decision = Decision(request.request_id, True, None, EV_RENEW,
                            lease.lease_id, now_ms,
                            {"renew_count": lease.renew_count})
        self.decisions[request.request_id] = decision
        return decision

    # ---------------------------------------------------------------- grant
    def _grant(self, request: Request, now_ms: int,
                lease_id: str | None = None) -> Decision:
        if lease_id is None:
            lease_id = self._new_lease_id(request.pool_id, request.key)
        lease = Lease(
            lease_id=lease_id, key=request.key, pool_id=request.pool_id,
            state=GRANTED, revision=request.revision, priority=request.priority,
            grant_ms=now_ms,
            expire_ms=now_ms + self.pools[request.pool_id].lease_ms)
        self.leases[lease_id] = lease
        self.active_index[(request.pool_id, request.key)] = lease_id
        self.key_location[request.key] = request.pool_id
        self._emit(EV_GRANT, now_ms, request_id=request.request_id,
                   key=request.key, pool_id=request.pool_id,
                   lease_id=lease_id, revision=request.revision,
                   priority=request.priority,
                   detail={"expire_ms": lease.expire_ms})
        decision = Decision(request.request_id, True, None, EV_GRANT,
                            lease_id, now_ms)
        self.decisions[request.request_id] = decision
        return decision

    # -------------------------------------------------------- preempt/rollback
    def _preempt_or_deny(self, request: Request, now_ms: int,
                         pool: Pool) -> Decision:
        candidates = [
            lease for lease in self._active_leases(request.pool_id)
            if lease.priority < request.priority and lease.key != request.key
        ]
        if not candidates:
            # Deterministic conflict reference: highest priority then lease id.
            reference = sorted(
                self._active_leases(request.pool_id),
                key=lambda item: (-item.priority, item.lease_id))[0]
            return self._deny(
                request, now_ms, reference.lease_id,
                "pool full and no lower-priority lease is preemptable")
        # Stable victim rule: lowest priority, nearest renewal boundary (older
        # boundary first), then lease id for a fully deterministic choice.
        victim = min(
            candidates,
            key=lambda item: (item.priority,
                              item.revoke_boundary_ms
                              if item.revoke_boundary_ms is not None
                              else item.grant_ms,
                              item.lease_id))
        edge = self.rollback_targets.get(victim.key)
        target_rev = edge["target_rev"] if edge and \
            edge["from_rev"] == victim.revision else None
        winner_lease_id = self._new_lease_id(request.pool_id, request.key)
        self._revoke(victim, request, now_ms, target_rev, winner_lease_id)
        # Capacity has been released safely; grant the winner immediately.
        return self._grant(request, now_ms, lease_id=winner_lease_id)

    def _revoke(self, victim: Lease, winner: Request, now_ms: int,
                target_rev: int | None, winner_lease_id: str) -> None:
        boundary = (victim.revoke_boundary_ms
                    if victim.revoke_boundary_ms is not None
                    else victim.grant_ms)
        if target_rev is not None:
            # Rollback converges every unexpired lease (including the victim)
            # and becomes the victim's terminal state; the revoke event still
            # records the safe boundary without losing renewals.
            self._rollback_key(victim.key, target_rev, winner.request_id,
                               boundary)
        else:
            victim.state = self._move(victim, "REVOKE")
        victim.revoked_by_lease_id = winner_lease_id
        self._emit(EV_REVOKE, boundary, request_id=winner.request_id,
                   key=victim.key, pool_id=victim.pool_id,
                   lease_id=victim.lease_id, revision=victim.revision,
                   priority=victim.priority,
                   detail={"boundary_ms": boundary,
                           "renew_count": victim.renew_count,
                           "winner_request_id": winner.request_id,
                           "rolled_back": target_rev is not None})
        self._release(victim)

    def _rollback_key(self, key: str, target_rev: int, trigger_request_id: str,
                      at_ms: int) -> None:
        # All still-active leases of the key converge to the target revision.
        for lease in sorted(self.leases.values(), key=lambda item: item.lease_id):
            if lease.key != key or not lease.active:
                continue
            lease.state = self._move(lease, "ROLLBACK")
            lease.rolledback_rev = target_rev
            self._emit(EV_ROLLBACK, at_ms,
                       request_id=trigger_request_id, key=key,
                       pool_id=lease.pool_id, lease_id=lease.lease_id,
                       revision=target_rev,
                       detail={"previous_rev": lease.revision})
            lease.revision = target_rev
            self._release(lease)

    # ------------------------------------------------------------------ deny
    def _deny(self, request: Request, now_ms: int, conflict_lease_id: str,
              reason: str) -> Decision:
        self._emit(EV_DENY, now_ms, request_id=request.request_id,
                   key=request.key, pool_id=request.pool_id,
                   lease_id=conflict_lease_id, revision=request.revision,
                   priority=request.priority, detail={"reason": reason})
        decision = Decision(request.request_id, False, E_LEASE, EV_DENY,
                            conflict_lease_id, now_ms, {"reason": reason,
                                                        "conflict_lease_id":
                                                            conflict_lease_id})
        self.decisions[request.request_id] = decision
        return decision

    # --------------------------------------------------------------- queries
    def snapshot_state(self) -> dict:
        return {
            "leases": {lease_id: lease.to_dict()
                       for lease_id, lease in sorted(self.leases.items())},
            "active_index": {f"{pool}|{key}:": lease_id
                             for (pool, key), lease_id
                             in sorted(self.active_index.items())},
            "last_event_seq": self._seq,
        }

    def rebuild_from_ledger(self, events: list[dict]) -> None:
        """Reconstruct lease bookkeeping from persisted ledger events.

        Used on recovery when no trustworthy snapshot exists.
        """
        for row in events:
            kind = row["event_type"]
            lease_id = row["lease_id"]
            if kind == EV_GRANT:
                lease = Lease(
                    lease_id=lease_id, key=row["key"],
                    pool_id=row["pool_id"], state=GRANTED,
                    revision=row["revision"], priority=row["priority"],
                    grant_ms=row["at_ms"],
                    expire_ms=row["detail"].get("expire_ms", row["at_ms"]))
                self.leases[lease_id] = lease
                self.active_index[(lease.pool_id, lease.key)] = lease_id
                self.key_location[lease.key] = lease.pool_id
            elif lease_id is None or lease_id not in self.leases:
                continue
            else:
                lease = self.leases[lease_id]
                if kind == EV_RENEW:
                    lease.state = RENEWED
                    lease.renew_count = row["detail"]["renew_count"]
                    lease.last_renew_ms = row["at_ms"]
                    lease.revoke_boundary_ms = row["at_ms"]
                    lease.expire_ms = row["detail"]["expire_ms"]
                    lease.revision = row["revision"]
                elif kind == EV_EXPIRE:
                    lease.state = "expired"
                    self._release(lease)
                elif kind == EV_REVOKE:
                    lease.state = "revoked"
                    self._release(lease)
                elif kind == EV_ROLLBACK:
                    lease.state = "rolledback"
                    lease.rolledback_rev = row["revision"]
                    self._release(lease)
            self._seq = max(self._seq, row["seq"])
