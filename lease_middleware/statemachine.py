"""Explicit lease generation state machine."""

from __future__ import annotations

from typing import Any

from .errors import LeaseError


GRANTED = "granted"
RENEWED = "renewed"
RELEASED = "released"
EXPIRED = "expired"

ACTIVE_STATES = {GRANTED, RENEWED}
TERMINAL_STATES = {RELEASED, EXPIRED}


ALLOWED_TRANSITIONS: dict[tuple[str | None, str], str] = {
    (None, "grant"): GRANTED,
    (GRANTED, "renew"): RENEWED,
    (RENEWED, "renew"): RENEWED,
    (GRANTED, "release"): RELEASED,
    (RENEWED, "release"): RELEASED,
    (GRANTED, "expire"): EXPIRED,
    (RENEWED, "expire"): EXPIRED,
}


def require_transition(from_state: str | None, trigger: str) -> str:
    key = (from_state, trigger)
    if key not in ALLOWED_TRANSITIONS:
        raise LeaseError(
            f"transition {trigger!r} is not allowed from state {from_state!r}",
            field="state",
            details={"from_state": from_state, "trigger": trigger},
        )
    return ALLOWED_TRANSITIONS[key]


def match_version_tag(
    *,
    lease_id: str,
    current_tag: int,
    requested_tag: int | None,
) -> int:
    """Validate the caller-supplied tag and return the matched tag."""

    if requested_tag is None:
        return current_tag
    if requested_tag == current_tag:
        return current_tag
    if requested_tag < current_tag:
        reason = "stale_tag_or_rollback"
        message = "request carries an expired/rolled-back generation tag"
    else:
        reason = "generation_skip"
        message = "request skips one or more lease generations"
    raise LeaseError(
        message,
        field="version_tag",
        details={
            "lease_id": lease_id,
            "current_tag": current_tag,
            "requested_tag": requested_tag,
            "reason": reason,
        },
    )


def transition_record(
    *,
    event_seq: int,
    lease: dict[str, Any],
    trigger: str,
    reason: str,
    at_ms: int,
    matched_tag: int | None,
    from_state: str | None,
    to_state: str,
    generation_after: int,
    expires_at_ms: int | None,
    generation_before: int | None = None,
) -> dict[str, Any]:
    if generation_before is None and from_state is not None:
        generation_before = lease.get("generation")
    return {
        "entry_kind": "lease_transition",
        "event_seq": event_seq,
        "lease_id": lease["lease_id"],
        "request_id": lease.get("request_id"),
        "equity_id": lease["equity_id"],
        "node_id": lease["node_id"],
        "node_generation": lease["node_generation"],
        "generation_before": generation_before,
        "generation_after": generation_after,
        "matched_tag": matched_tag,
        "version_tag": generation_after,
        "from_state": from_state,
        "to_state": to_state,
        "trigger": trigger,
        "reason": reason,
        "at_ms": at_ms,
        "expires_at_ms": expires_at_ms,
    }
