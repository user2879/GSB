from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import LeaseError, StaleError


STATES = ("new", "granted", "renewed", "released", "expired")
ACTIVE_STATES = frozenset({"granted", "renewed"})
_TERMINAL_REGRANT_STATES = frozenset({"released", "expired"})

# This table is the complete authority for legal transitions.  Membership is
# explicit; no decision is derived from dictionary or hash iteration order.
ALLOWED_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("new", "granted"),
        ("granted", "renewed"),
        ("granted", "released"),
        ("granted", "expired"),
        ("renewed", "renewed"),
        ("renewed", "released"),
        ("renewed", "expired"),
        ("released", "granted"),
        ("expired", "granted"),
    }
)


@dataclass
class LeaseState:
    lease_id: str
    node_id: str
    lease_type: str
    ttl_ms: int
    renew_interval_ms: int
    quota_version: int
    state: str = "new"
    label: int = 0
    expires_at_ms: int | None = None
    global_slot: int | None = None
    node_slot: int | None = None
    owner: str | None = None
    last_event_seq: int = 0
    last_reason: str = "new"

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "lease_type": self.lease_type,
            "ttl_ms": self.ttl_ms,
            "renew_interval_ms": self.renew_interval_ms,
            "quota_version": self.quota_version,
            "state": self.state,
            "label": self.label,
            "expires_at_ms": self.expires_at_ms,
            "global_slot": self.global_slot,
            "node_slot": self.node_slot,
            "owner": self.owner,
            "last_event_seq": self.last_event_seq,
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "LeaseState":
        record = cls(
            lease_id=payload["lease_id"],
            node_id=payload["node_id"],
            lease_type=payload["lease_type"],
            ttl_ms=payload["ttl_ms"],
            renew_interval_ms=payload["renew_interval_ms"],
            quota_version=payload["quota_version"],
            state=payload["state"],
            label=payload["label"],
            expires_at_ms=payload.get("expires_at_ms"),
            global_slot=payload.get("global_slot"),
            node_slot=payload.get("node_slot"),
            owner=payload.get("owner"),
            last_event_seq=payload.get("last_event_seq", 0),
            last_reason=payload.get("last_reason", "new"),
        )
        if record.state not in STATES:
            raise LeaseError("snapshot contains an unknown lease state", lease_id=record.lease_id)
        return record


def transition(
    lease: LeaseState,
    target_state: str,
    reason: str,
    expected_label: int,
    event_seq: int,
    at_ms: int,
    *,
    new_label: int | None = None,
) -> dict[str, Any]:
    """Apply one accepted transition and return its immutable audit record.

    ``new_label`` exists only for explicit verifier/state-machine tests.  Normal
    callers pass ``None``; the sole legal next label is current + 1.
    """

    if expected_label != lease.label:
        raise StaleError(
            "request label is stale",
            lease_id=lease.lease_id,
            expected_label=expected_label,
            current_label=lease.label,
            state=lease.state,
        )
    candidate_label = lease.label + 1 if new_label is None else new_label
    if candidate_label != lease.label + 1:
        raise LeaseError(
            "lease algebra can only advance by exactly one generation",
            lease_id=lease.lease_id,
            current_label=lease.label,
            requested_label=candidate_label,
        )
    if target_state not in STATES:
        raise LeaseError("unknown target lease state", lease_id=lease.lease_id, target_state=target_state)
    if (lease.state, target_state) not in ALLOWED_TRANSITIONS:
        raise LeaseError(
            "lease transition is not permitted",
            lease_id=lease.lease_id,
            from_state=lease.state,
            to_state=target_state,
            allowed=sorted(f"{a}->{b}" for a, b in ALLOWED_TRANSITIONS),
        )
    label_before = lease.label
    state_before = lease.state
    lease.state = target_state
    lease.label = candidate_label
    lease.last_event_seq = event_seq
    lease.last_reason = reason
    return {
        "event_seq": event_seq,
        "lease_id": lease.lease_id,
        "at_ms": at_ms,
        "reason": reason,
        "from_state": state_before,
        "to_state": target_state,
        "label_before": label_before,
        "label_after": candidate_label,
    }


@dataclass
class SlotPool:
    global_limit: int
    node_limits: dict[str, int]
    global_occupied: dict[int, str] = field(default_factory=dict)
    node_occupied: dict[str, dict[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for node_id in sorted(self.node_limits):
            self.node_occupied.setdefault(node_id, {})

    @property
    def global_used(self) -> int:
        return len(self.global_occupied)

    def node_used(self, node_id: str) -> int:
        return len(self.node_occupied[node_id])

    def remaining_global(self) -> int:
        return self.global_limit - self.global_used

    def acquire(self, lease_id: str, node_id: str) -> tuple[int, int] | None:
        if node_id not in self.node_limits:
            raise LeaseError("lease references an unknown node", lease_id=lease_id, node_id=node_id)
        global_slot = self._first_free(self.global_limit, self.global_occupied)
        node_slot = self._first_free(self.node_limits[node_id], self.node_occupied[node_id])
        if global_slot is None or node_slot is None:
            return None
        self.global_occupied[global_slot] = lease_id
        self.node_occupied[node_id][node_slot] = lease_id
        return global_slot, node_slot

    def release(self, lease_id: str, node_id: str, global_slot: int | None, node_slot: int | None) -> None:
        if global_slot is not None:
            owner = self.global_occupied.get(global_slot)
            if owner != lease_id:
                raise LeaseError(
                    "global quota slot ownership mismatch",
                    lease_id=lease_id,
                    global_slot=global_slot,
                    owner=owner,
                )
            del self.global_occupied[global_slot]
        if node_slot is not None:
            owner = self.node_occupied.get(node_id, {}).get(node_slot)
            if owner != lease_id:
                raise LeaseError(
                    "node quota slot ownership mismatch",
                    lease_id=lease_id,
                    node_id=node_id,
                    node_slot=node_slot,
                    owner=owner,
                )
            del self.node_occupied[node_id][node_slot]

    def assert_consistent(self, leases: dict[str, LeaseState]) -> None:
        if len(self.global_occupied) > self.global_limit:
            raise LeaseError("global quota exceeded")
        seen_global: set[int] = set()
        for lease_id in sorted(self.global_occupied):
            slot = lease_id
            owner = self.global_occupied[slot]
            record = leases.get(owner)
            if record is None or record.global_slot != slot:
                raise LeaseError("orphaned global slot", global_slot=slot, owner=owner)
            if slot in seen_global:
                raise LeaseError("global slot shared by more than one lease", global_slot=slot)
            seen_global.add(slot)
        for node_id in sorted(self.node_occupied):
            used = len(self.node_occupied[node_id])
            if used > self.node_limits[node_id]:
                raise LeaseError("per-node quota exceeded", node_id=node_id)
            seen_node: set[int] = set()
            for slot in sorted(self.node_occupied[node_id]):
                owner = self.node_occupied[node_id][slot]
                record = leases.get(owner)
                if record is None or record.node_slot != slot or record.node_id != node_id:
                    raise LeaseError(
                        "orphaned node slot",
                        node_id=node_id,
                        node_slot=slot,
                        owner=owner,
                    )
                if slot in seen_node:
                    raise LeaseError("node slot shared by more than one lease", node_id=node_id, node_slot=slot)
                seen_node.add(slot)
        for lease_id in sorted(leases):
            record = leases[lease_id]
            if record.active:
                if record.global_slot not in self.global_occupied:
                    raise LeaseError("active lease does not own a global slot", lease_id=lease_id)
                if record.node_slot not in self.node_occupied[record.node_id]:
                    raise LeaseError("active lease does not own a node slot", lease_id=lease_id)
                if self.global_occupied[record.global_slot] != lease_id:
                    raise LeaseError("active lease global slot has another owner", lease_id=lease_id)
                if self.node_occupied[record.node_id][record.node_slot] != lease_id:
                    raise LeaseError("active lease node slot has another owner", lease_id=lease_id)
            else:
                if record.global_slot is not None or record.node_slot is not None:
                    raise LeaseError("inactive lease still owns a quota slot", lease_id=lease_id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "global_limit": self.global_limit,
            "node_limits": {key: self.node_limits[key] for key in sorted(self.node_limits)},
            "global_occupied": [
                {"slot": slot, "lease_id": self.global_occupied[slot]}
                for slot in sorted(self.global_occupied)
            ],
            "node_occupied": {
                node_id: [
                    {"slot": slot, "lease_id": self.node_occupied[node_id][slot]}
                    for slot in sorted(self.node_occupied[node_id])
                ]
                for node_id in sorted(self.node_occupied)
            },
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "SlotPool":
        pool = cls(payload["global_limit"], dict(payload["node_limits"]))
        for item in payload.get("global_occupied", []):
            pool.global_occupied[int(item["slot"])] = item["lease_id"]
        for node_id, items in payload.get("node_occupied", {}).items():
            pool.node_occupied.setdefault(node_id, {})
            for item in items:
                pool.node_occupied[node_id][int(item["slot"])] = item["lease_id"]
        return pool

    @staticmethod
    def _first_free(limit: int, occupied: dict[int, str]) -> int | None:
        for slot in range(1, limit + 1):
            if slot not in occupied:
                return slot
        return None
