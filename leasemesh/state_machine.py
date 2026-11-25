from __future__ import annotations

from typing import Any, Optional

from .errors import E_STATE, MeshError


GRANTED = "granted"
RENEWED = "renewed"
RELEASED = "released"
EXPIRED = "expired"

ACTIVE_STATES = {GRANTED, RENEWED}
TERMINAL_STATES = {RELEASED, EXPIRED}

PERMITTED_TRANSITIONS: dict[tuple[Optional[str], str], set[str]] = {
    (None, GRANTED): {"grant", "lease_granted"},
    (GRANTED, RENEWED): {"renew", "lease_renewed"},
    (RENEWED, RENEWED): {"renew", "lease_renewed"},
    (GRANTED, RELEASED): {"release", "lease_released"},
    (RENEWED, RELEASED): {"release", "lease_released"},
    (GRANTED, EXPIRED): {"expire", "ttl_expiry", "expiration_reclaim", "quota_fenced"},
    (RENEWED, EXPIRED): {"expire", "ttl_expiry", "expiration_reclaim", "quota_fenced"},
}

KIND_TARGET = {
    "lease_granted": GRANTED,
    "lease_renewed": RENEWED,
    "lease_released": RELEASED,
    "lease_expired": EXPIRED,
}


def assert_transition(before: Optional[str], after: str, reason: str) -> None:
    allowed_reasons = PERMITTED_TRANSITIONS.get((before, after), set())
    if reason not in allowed_reasons:
        raise MeshError(
            f"illegal lease transition {before} -> {after} for reason {reason}",
            code=E_STATE,
            details={"before_state": before, "after_state": after, "reason": reason},
        )


def transition_record(
    *,
    lease_id: str,
    before_state: Optional[str],
    after_state: str,
    reason: str,
    generation_before: Optional[int],
    generation_after: int,
    event_seq: Optional[int] = None,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    assert_transition(before_state, after_state, reason)
    if generation_before is not None and int(generation_after) < int(generation_before):
        raise MeshError(
            "lease generation moved backwards",
            code="E_LEASE",
            details={"before": generation_before, "after": generation_after},
        )
    return {
        "lease_id": lease_id,
        "before_state": before_state,
        "after_state": after_state,
        "reason": reason,
        "generation_before": generation_before,
        "generation_after": int(generation_after),
        "event_seq": event_seq,
        "request_id": request_id,
    }
