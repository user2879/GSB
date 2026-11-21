"""Deterministic lease ownership state machine."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import WATERMARK_SCALE, WATERMARK_TRUNCATION, watermark_fixed
from .errors import LeaseError
from .models import ClusterSpec, LeaseEvent


ACTIVE_STATES = frozenset({"granted", "renewed"})
TRANSITION_STATES = frozenset({"granted", "renewed", "released", "expired"})


@dataclass
class ActiveLease:
    holder_id: str
    state: str
    priority: int
    started_ms: int
    last_renewed_ms: int
    expires_ms: int
    renewal_count: int
    last_event_seq: int
    provenance: str
    real_deadline_ns: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "expires_ms": self.expires_ms,
            "holder_id": self.holder_id,
            "last_event_seq": self.last_event_seq,
            "last_renewed_ms": self.last_renewed_ms,
            "priority": self.priority,
            "provenance": self.provenance,
            "renewal_count": self.renewal_count,
            "started_ms": self.started_ms,
            "state": self.state,
        }

    @classmethod
    def from_restore(cls, raw: dict[str, Any], lease_duration_ms: int) -> "ActiveLease":
        holder_id = raw["holder_id"]
        state = raw.get("state", "granted")
        if state not in ACTIVE_STATES:
            raise LeaseError("E_REPLY", f"invalid restored state for {holder_id}: {state}", field="leases.state")
        started_ms = int(raw.get("started_ms", 0))
        last_renewed_ms = int(raw.get("last_renewed_ms", started_ms))
        renewal_count = int(raw.get("renewal_count", 0))
        return cls(
            holder_id=holder_id,
            state=state,
            priority=int(raw.get("priority", 0)),
            started_ms=started_ms,
            last_renewed_ms=last_renewed_ms,
            expires_ms=last_renewed_ms + lease_duration_ms,
            renewal_count=renewal_count,
            last_event_seq=int(raw.get("last_event_seq", 0)),
            provenance="recovery_recomputed",
        )


class ArbitrationEngine:
    """Pure deterministic state machine with optional real-clock expiry guard.

    - logical: ownership is decided only by event timestamps (used by replay).
    - real: ownership is decided only by the server monotonic clock.
    - hybrid: either logical expiry or real-clock expiry expires a lease.
    """

    def __init__(
        self,
        spec: ClusterSpec,
        priorities: dict[str, int],
        parameter_version: int = 1,
        *,
        clock_mode: str = "hybrid",
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if clock_mode not in {"logical", "real", "hybrid"}:
            raise ValueError(f"unsupported clock_mode: {clock_mode}")
        self.spec = spec
        self.priorities = dict(priorities)
        self.parameter_version = parameter_version
        self.clock_mode = clock_mode
        self.monotonic_ns = monotonic_ns or time.monotonic_ns
        self.active: dict[str, ActiveLease] = {}
        self.last_timestamp_ms: int | None = None
        self.last_event_seq = 0
        self._cache: dict[int, dict[str, Any]] = {}

    def _real_now(self) -> int:
        return self.monotonic_ns()

    def _real_deadline(self, duration_ms: int) -> int:
        return self._real_now() + duration_ms * 1_000_000

    def _active_sorted(self) -> list[ActiveLease]:
        return sorted(self.active.values(), key=lambda lease: lease.holder_id)

    def _watermark(self) -> int:
        return watermark_fixed(len(self.active), self.spec.capacity)

    def _transition(
        self,
        lease: ActiveLease,
        *,
        before: str,
        after: str,
        reason: str,
        event_seq: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        return {
            "active_count": len(self.active),
            "after_state": after,
            "before_state": before,
            "cluster_id": self.spec.cluster_id,
            "event_seq": event_seq,
            "holder_id": lease.holder_id,
            "reason": reason,
            "timestamp_ms": timestamp_ms,
        }

    def _expire(self, timestamp_ms: int, event_seq: int) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        now_ns = self._real_now()
        for holder_id in sorted(list(self.active)):
            lease = self.active[holder_id]
            logical_due = (
                self.clock_mode in {"logical", "hybrid"}
                and timestamp_ms >= lease.expires_ms
            )
            real_due = (
                self.clock_mode in {"real", "hybrid"}
                and lease.real_deadline_ns > 0
                and now_ns >= lease.real_deadline_ns
            )
            if not (logical_due or real_due):
                continue
            del self.active[holder_id]
            transitions.append(
                self._transition(
                    lease,
                    before=lease.state,
                    after="expired",
                    reason="lease_expired",
                    event_seq=event_seq,
                    timestamp_ms=timestamp_ms,
                )
            )
        return transitions

    def _grant(
        self,
        holder_id: str,
        *,
        timestamp_ms: int,
        event_seq: int,
        state: str,
        reason: str,
        transition_before: str | None = None,
    ) -> tuple[ActiveLease, dict[str, Any]]:
        priority = self.priorities[holder_id]
        lease = ActiveLease(
            holder_id=holder_id,
            state=state,
            priority=priority,
            started_ms=timestamp_ms,
            last_renewed_ms=timestamp_ms,
            expires_ms=timestamp_ms + self.spec.lease_duration_ms,
            renewal_count=0,
            last_event_seq=event_seq,
            provenance="arbitration_service",
            real_deadline_ns=self._real_deadline(self.spec.lease_duration_ms),
        )
        self.active[holder_id] = lease
        transition = self._transition(
            lease,
            before=transition_before or "absent",
            after=state,
            reason=reason,
            event_seq=event_seq,
            timestamp_ms=timestamp_ms,
        )
        return lease, transition

    def _base_response(
        self,
        *,
        event_seq: int,
        holder_id: str,
        event_type: str,
        timestamp_ms: int,
        decision: str,
        reason: str,
        transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "active_holders": [lease.to_json() for lease in self._active_sorted()],
            "capacity": self.spec.capacity,
            "cluster_id": self.spec.cluster_id,
            "decision": decision,
            "event_seq": event_seq,
            "event_type": event_type,
            "holder_id": holder_id,
            "ok": True,
            "parameter_version": self.parameter_version,
            "reason": reason,
            "timestamp_ms": timestamp_ms,
            "transitions": transitions,
            "type": "decision",
            "used_capacity": len(self.active),
            "watermark_fixed": self._watermark(),
            "watermark_scale": WATERMARK_SCALE,
            "watermark_truncation": WATERMARK_TRUNCATION,
        }

    def apply(
        self,
        *,
        holder_id: str,
        event_seq: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        if holder_id not in self.priorities:
            transitions = self._expire(timestamp_ms, event_seq)
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="apply",
                timestamp_ms=timestamp_ms,
                decision="rejected",
                reason="unknown_holder",
                transitions=transitions,
            )
        transitions = self._expire(timestamp_ms, event_seq)
        if holder_id in self.active:
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="apply",
                timestamp_ms=timestamp_ms,
                decision="rejected",
                reason="already_holder",
                transitions=transitions,
            )

        if len(self.active) < self.spec.capacity:
            _, transition = self._grant(
                holder_id,
                timestamp_ms=timestamp_ms,
                event_seq=event_seq,
                state="granted",
                reason="capacity_available",
            )
            transitions.append(transition)
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="apply",
                timestamp_ms=timestamp_ms,
                decision="granted",
                reason="capacity_available",
                transitions=transitions,
            )

        candidate_priority = self.priorities[holder_id]
        victim = min(
            self.active.values(),
            key=lambda lease: (lease.priority, lease.holder_id),
        )
        if candidate_priority <= victim.priority:
            reason = "capacity_full_priority_tie" if candidate_priority == victim.priority else "capacity_full_lower_priority"
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="apply",
                timestamp_ms=timestamp_ms,
                decision="rejected",
                reason=reason,
                transitions=transitions,
            )

        del self.active[victim.holder_id]
        transitions.append(
            self._transition(
                victim,
                before=victim.state,
                after="expired",
                reason="higher_priority_takeover",
                event_seq=event_seq,
                timestamp_ms=timestamp_ms,
            )
        )
        _, transition = self._grant(
            holder_id,
            timestamp_ms=timestamp_ms,
            event_seq=event_seq,
            state="granted",
            reason="higher_priority_takeover",
        )
        transitions.append(transition)
        return self._base_response(
            event_seq=event_seq,
            holder_id=holder_id,
            event_type="apply",
            timestamp_ms=timestamp_ms,
            decision="granted",
            reason="higher_priority_takeover",
            transitions=transitions,
        )

    def renew(
        self,
        *,
        holder_id: str,
        event_seq: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        transitions = self._expire(timestamp_ms, event_seq)
        lease = self.active.get(holder_id)
        if lease is None:
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="renew",
                timestamp_ms=timestamp_ms,
                decision="rejected",
                reason="not_holder_or_expired",
                transitions=transitions,
            )
        before = lease.state
        lease.state = "renewed"
        lease.last_renewed_ms = timestamp_ms
        lease.expires_ms = timestamp_ms + self.spec.lease_duration_ms
        lease.renewal_count += 1
        lease.last_event_seq = event_seq
        lease.provenance = "true_renewal"
        lease.real_deadline_ns = self._real_deadline(self.spec.lease_duration_ms)
        transitions.append(
            self._transition(
                lease,
                before=before,
                after="renewed",
                reason="renewal_accepted",
                event_seq=event_seq,
                timestamp_ms=timestamp_ms,
            )
        )
        return self._base_response(
            event_seq=event_seq,
            holder_id=holder_id,
            event_type="renew",
            timestamp_ms=timestamp_ms,
            decision="renewed",
            reason="renewal_accepted",
            transitions=transitions,
        )

    def release(
        self,
        *,
        holder_id: str,
        event_seq: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        transitions = self._expire(timestamp_ms, event_seq)
        lease = self.active.get(holder_id)
        if lease is None:
            return self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type="release",
                timestamp_ms=timestamp_ms,
                decision="released",
                reason="no_active_lease",
                transitions=transitions,
            )
        before = lease.state
        del self.active[holder_id]
        transitions.append(
            self._transition(
                lease,
                before=before,
                after="released",
                reason="holder_released",
                event_seq=event_seq,
                timestamp_ms=timestamp_ms,
            )
        )
        return self._base_response(
            event_seq=event_seq,
            holder_id=holder_id,
            event_type="release",
            timestamp_ms=timestamp_ms,
            decision="released",
            reason="holder_released",
            transitions=transitions,
        )

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        event_type = request.get("event_type")
        event_seq = request.get("event_seq")
        if not isinstance(event_seq, int) or isinstance(event_seq, bool) or event_seq <= 0:
            raise LeaseError("E_PROTOCOL", "event_seq must be a positive integer", field="event_seq")
        if event_seq in self._cache:
            return copy.deepcopy(self._cache[event_seq])
        timestamp_ms = request.get("timestamp_ms")
        if not isinstance(timestamp_ms, int) or isinstance(timestamp_ms, bool) or timestamp_ms < 0:
            raise LeaseError("E_PROTOCOL", "timestamp_ms must be a non-negative integer", field="timestamp_ms")
        holder_id = request.get("holder_id")
        if not isinstance(holder_id, str) or not holder_id:
            raise LeaseError("E_PROTOCOL", "holder_id is required", field="holder_id")
        if self.last_timestamp_ms is not None and timestamp_ms < self.last_timestamp_ms:
            response = self._base_response(
                event_seq=event_seq,
                holder_id=holder_id,
                event_type=str(event_type),
                timestamp_ms=timestamp_ms,
                decision="rejected",
                reason="timestamp_regression",
                transitions=[],
            )
        elif event_type == "apply":
            response = self.apply(holder_id=holder_id, event_seq=event_seq, timestamp_ms=timestamp_ms)
        elif event_type == "renew":
            response = self.renew(holder_id=holder_id, event_seq=event_seq, timestamp_ms=timestamp_ms)
        elif event_type == "release":
            response = self.release(holder_id=holder_id, event_seq=event_seq, timestamp_ms=timestamp_ms)
        else:
            raise LeaseError("E_PROTOCOL", f"unsupported event_type: {event_type}", field="event_type")
        self.last_timestamp_ms = timestamp_ms
        self.last_event_seq = max(self.last_event_seq, event_seq)
        self._cache[event_seq] = copy.deepcopy(response)
        return response

    def status(self, timestamp_ms: int | None = None) -> dict[str, Any]:
        if timestamp_ms is None:
            timestamp_ms = self.last_timestamp_ms or 0
        transitions = self._expire(timestamp_ms, self.last_event_seq)
        return {
            "active_holders": [lease.to_json() for lease in self._active_sorted()],
            "capacity": self.spec.capacity,
            "cluster_id": self.spec.cluster_id,
            "ok": True,
            "parameter_version": self.parameter_version,
            "timestamp_ms": timestamp_ms,
            "transitions": transitions,
            "type": "status",
            "used_capacity": len(self.active),
            "watermark_fixed": self._watermark(),
            "watermark_scale": WATERMARK_SCALE,
            "watermark_truncation": WATERMARK_TRUNCATION,
        }

    def restore(
        self,
        raw_leases: list[dict[str, Any]],
        last_event_seq: int,
        last_timestamp_ms: int = 0,
    ) -> dict[str, Any]:
        """Restore committed active leases and recompute deadlines from ledger duration."""

        self.active.clear()
        self._cache.clear()
        self.last_event_seq = last_event_seq
        now_ns = self._real_now()
        continuations: list[dict[str, Any]] = []
        logical_anchor = last_timestamp_ms
        for raw in sorted(raw_leases, key=lambda item: item["holder_id"]):
            lease = ActiveLease.from_restore(raw, self.spec.lease_duration_ms)
            recorded_expires = int(raw.get("expires_ms", lease.expires_ms))
            remaining = max(0, lease.expires_ms - logical_anchor)
            if remaining <= 0:
                continue
            lease.real_deadline_ns = now_ns + remaining * 1_000_000
            lease.priority = int(raw.get("priority", self.priorities.get(lease.holder_id, 0)))
            self.active[lease.holder_id] = lease
            continuation = (
                "true_renewal_continuation"
                if lease.renewal_count > 0
                else "recovery_recomputed_continuation"
            )
            lease.provenance = "recovery_recomputed"
            continuations.append(
                {
                    "continuation": continuation,
                    "holder_id": lease.holder_id,
                    "last_renewed_ms": lease.last_renewed_ms,
                    "recorded_expires_ms": recorded_expires,
                    "recomputed_expires_ms": lease.expires_ms,
                    "renewal_count": lease.renewal_count,
                }
            )
        self.last_timestamp_ms = last_timestamp_ms if raw_leases or last_event_seq > 0 else None
        return {
            "active_holders": [lease.to_json() for lease in self._active_sorted()],
            "capacity": self.spec.capacity,
            "cluster_id": self.spec.cluster_id,
            "continuations": continuations,
            "last_event_seq": last_event_seq,
            "ok": True,
            "parameter_version": self.parameter_version,
            "type": "restored",
            "used_capacity": len(self.active),
            "watermark_fixed": self._watermark(),
            "watermark_scale": WATERMARK_SCALE,
        }


def event_to_request(event: LeaseEvent) -> dict[str, Any]:
    return {
        "cluster_id": event.cluster_id,
        "event_seq": event.event_seq,
        "event_type": event.event_type,
        "holder_id": event.holder_id,
        "timestamp_ms": event.timestamp_ms,
        "type": "lease",
    }
