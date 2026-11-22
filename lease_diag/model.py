"""Core domain model: pools, requests, leases, events and the permission table.

Lease lifecycle states are ``granted`` -> ``renewed`` (repeatable) and then
one of the terminal states ``expired`` / ``revoked`` / ``rolledback``.
Every legal transition is listed explicitly in ``TRANSITIONS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

GRANTED = "granted"
RENEWED = "renewed"
EXPIRED = "expired"
REVOKED = "revoked"
ROLLEDBACK = "rolledback"

ACTIVE_STATES = frozenset({GRANTED, RENEWED})
TERMINAL_STATES = frozenset({EXPIRED, REVOKED, ROLLEDBACK})

# Explicit state-machine permission table: from_state -> {event -> to_state}.
TRANSITIONS: dict[str, dict[str, str]] = {
    GRANTED: {
        "RENEW": RENEWED,
        "EXPIRE": EXPIRED,
        "REVOKE": REVOKED,
        "ROLLBACK": ROLLEDBACK,
    },
    RENEWED: {
        "RENEW": RENEWED,
        "EXPIRE": EXPIRED,
        "REVOKE": REVOKED,
        "ROLLBACK": ROLLEDBACK,
    },
    EXPIRED: {},
    REVOKED: {},
    ROLLEDBACK: {},
}

# Event kinds persisted into the ledger. Replayable events are reproduced by
# the deterministic engine; QUEUE is a scheduler observation only.
EV_GRANT = "GRANT"
EV_RENEW = "RENEW"
EV_EXPIRE = "EXPIRE"
EV_REVOKE = "REVOKE"
EV_ROLLBACK = "ROLLBACK"
EV_QUEUE = "QUEUE"
EV_DENY = "DENY"
REPLAYABLE_EVENTS = frozenset(
    {EV_GRANT, EV_RENEW, EV_EXPIRE, EV_REVOKE, EV_ROLLBACK, EV_DENY})


@dataclass(frozen=True)
class Pool:
    pool_id: str
    capacity: int
    lease_ms: int
    renew_ms: int
    priority: int
    quota: int          # max concurrent in-flight grants for this pool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RollbackEdge:
    key: str
    from_rev: int
    target_rev: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Request:
    request_id: str
    key: str
    pool_id: str
    revision: int
    priority: int
    arrive_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Lease:
    lease_id: str
    key: str
    pool_id: str
    state: str
    revision: int
    priority: int
    grant_ms: int
    expire_ms: int
    renew_count: int = 0
    last_renew_ms: int | None = None
    # The safe revocation boundary is the most recent committed renewal.
    revoke_boundary_ms: int | None = None
    revoked_by_lease_id: str | None = None
    rolledback_rev: int | None = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    seq: int
    event_type: str
    request_id: str | None
    key: str | None
    pool_id: str | None
    lease_id: str | None
    at_ms: int
    revision: int | None = None
    priority: int | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """Outcome of adjudicating one request."""

    request_id: str
    accepted: bool
    code: str | None
    event_type: str
    lease_id: str | None
    at_ms: int
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def transition_allowed(from_state: str, action: str) -> bool:
    return action in TRANSITIONS.get(from_state, {})


def next_state(from_state: str, action: str) -> str:
    table = TRANSITIONS.get(from_state)
    if not table or action not in table:
        raise KeyError(f"illegal transition {from_state} --{action}-->")
    return table[action]
