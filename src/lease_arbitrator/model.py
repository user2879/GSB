from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


STATE_DECLARED = "declared"
STATE_QUEUED = "queued"
STATE_GRANTED = "granted"
STATE_RENEWED = "renewed"
STATE_RELEASED = "released"
STATE_EXPIRED = "expired"
STATE_REVOKED = "revoked"
STATE_FAILED = "failed"

ACTIVE_STATES = {STATE_GRANTED, STATE_RENEWED}
TERMINAL_STATES = {STATE_RELEASED, STATE_EXPIRED, STATE_REVOKED, STATE_FAILED}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATE_DECLARED: {STATE_QUEUED, STATE_REVOKED},
    STATE_QUEUED: {STATE_GRANTED, STATE_REVOKED, STATE_FAILED},
    STATE_GRANTED: {
        STATE_RENEWED,
        STATE_RELEASED,
        STATE_EXPIRED,
        STATE_REVOKED,
    },
    STATE_RENEWED: {
        STATE_RENEWED,
        STATE_RELEASED,
        STATE_EXPIRED,
        STATE_REVOKED,
    },
    STATE_RELEASED: set(),
    STATE_EXPIRED: set(),
    STATE_REVOKED: set(),
    STATE_FAILED: {STATE_QUEUED},
}


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def transition_allowed(before: str, after: str) -> bool:
    return after in ALLOWED_TRANSITIONS.get(before, set())


@dataclass
class Transition:
    lease_id: str
    from_state: str
    to_state: str
    reason: str
    monotonic_ms: int
    event_seq: int | None = None
    generation_used: int | None = None
    generation_before: int | None = None
    generation_after: int | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "monotonic_ms": self.monotonic_ms,
            "event_seq": self.event_seq,
            "generation_used": self.generation_used,
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
            "error_code": self.error_code,
        }


@dataclass
class LeaseRuntime:
    lease_id: str
    node_id: str
    units: int
    priority: int
    hold_ms: int
    arrival_seq: int | None = None
    quota_version: int = 1
    references: list[str] = field(default_factory=list)
    state: str = STATE_DECLARED
    generation_current: int = 0
    charged_units: int = 0
    active_slot: bool = False
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    granted_monotonic_ms: int | None = None
    deadline_monotonic_ms: int | None = None
    last_event_seq: int | None = None
    history: list[Transition] = field(default_factory=list)
    stale: bool = False
    ownership_version: int = 1

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def wait_ms(self) -> int | None:
        if self.wait_start_ms is None or self.wait_end_ms is None:
            return None
        return max(0, self.wait_end_ms - self.wait_start_ms)

    def public_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "units": self.units,
            "priority": self.priority,
            "hold_ms": self.hold_ms,
            "arrival_seq": self.arrival_seq,
            "quota_version": self.quota_version,
            "references": self.references,
            "state": self.state,
            "generation_current": self.generation_current,
            "charged_units": self.charged_units,
            "active_slot": self.active_slot,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "wait_ms": self.wait_ms,
            "granted_monotonic_ms": self.granted_monotonic_ms,
            "deadline_monotonic_ms": self.deadline_monotonic_ms,
            "last_event_seq": self.last_event_seq,
            "stale": self.stale,
            "ownership_version": self.ownership_version,
            "history": [item.to_dict() for item in self.history],
        }
