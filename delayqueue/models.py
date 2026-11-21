"""Domain models, the explicit state-machine permission table and clocks."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

# --- Message states -------------------------------------------------------
PENDING = "pending"
DELAYED = "delayed"
READY = "ready"
INFLIGHT = "inflight"
ACKED = "acked"
DEAD = "dead"

STATES = (PENDING, DELAYED, READY, INFLIGHT, ACKED, DEAD)

# Recovery marker.  "unknown" is not a state machine state: it annotates
# inflight leases whose resolution was never committed before a hard kill.
UNKNOWN_MARKER = "unknown"

# --- Explicit transition permission table ---------------------------------
# (from_state, reason) -> to_state
TRANSITIONS: dict[tuple[str, str], str] = {
    (PENDING, "enqueue"): DELAYED,        # due time in the future
    (PENDING, "enqueue_now"): READY,      # due time already reached
    (DELAYED, "due"): READY,
    (DELAYED, "retry_backoff"): READY,    # backoff elapsed -> ready
    (READY, "claim"): INFLIGHT,
    (INFLIGHT, "ack"): ACKED,
    (INFLIGHT, "nack_retry"): DELAYED,
    (INFLIGHT, "nack_dead"): DEAD,
    # Visibility-timeout special case: no delivery-count increment, the
    # existing visibility record is retained (only closed, never deleted).
    (INFLIGHT, "visibility_timeout"): READY,
}


def next_state(from_state: str, reason: str) -> str:
    """Return the target state for an allowed transition or raise."""
    from .errors import IllegalTransitionError
    try:
        return TRANSITIONS[(from_state, reason)]
    except KeyError:
        raise IllegalTransitionError(
            f"transition not permitted",
            from_state=from_state, reason=reason,
        )


# --- Clocks ---------------------------------------------------------------
def wall_ms() -> int:
    """Wall clock epoch milliseconds (absolute scheduling time)."""
    return int(time.time() * 1000)


def mono_ms() -> int:
    """Monotonic milliseconds; the only basis for lease/timeout decisions."""
    return time.monotonic_ns() // 1_000_000


def payload_digest(payload: bytes) -> str:
    """Stable summary of an integer byte payload."""
    return hashlib.sha256(payload).hexdigest()


# --- Declaration / message data classes -----------------------------------
@dataclass(frozen=True)
class TopicSpec:
    topic: str
    visibility_timeout_ms: int
    max_deliveries: int
    backoff_ms: tuple[int, ...]
    concurrency: int


@dataclass(frozen=True)
class QueueSpec:
    global_concurrency: int
    topics: dict[str, TopicSpec]
    spec_version: str

    def canonical_digest(self) -> str:
        import json
        canonical = json.dumps(
            {
                "spec_version": self.spec_version,
                "global_concurrency": self.global_concurrency,
                "topics": {
                    name: {
                        "visibility_timeout_ms": t.visibility_timeout_ms,
                        "max_deliveries": t.max_deliveries,
                        "backoff_ms": list(t.backoff_ms),
                        "concurrency": t.concurrency,
                    }
                    for name, t in sorted(self.topics.items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass
class MessageInput:
    message_id: str
    topic: str
    deliver_at: int          # non-negative offset ms from queue epoch
    payload: bytes
    idempotency_key: str
    line_no: int = 0


@dataclass
class Message:
    message_id: str
    topic: str
    deliver_at_abs: int      # absolute wall-ms the message first becomes due
    origin_offset_ms: int    # original deliver_at offset as declared
    payload: bytes
    payload_summary: str
    idempotency_key: str
    state: str = PENDING
    attempts: int = 0
    suppressed: int = 0
    unknown_count: int = 0
    ready_monotonic_ms: int = 0
    last_error: str = ""
    first_error: str = ""
    seq: int = 0


@dataclass
class Delivery:
    """A leased message handed to a consumer thread."""

    message_id: str
    topic: str
    payload: bytes
    idempotency_key: str
    attempt: int
    lease_monotonic_ms: int
    visibility_timeout_ms: int
    unknown_redelivery: bool


@dataclass
class RecoveryReport:
    recovered_messages: int = 0
    unknown_messages: int = 0
    dead_messages: int = 0
    redelivered_after_visibility: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "recovered_messages": self.recovered_messages,
            "unknown_messages": self.unknown_messages,
            "dead_messages": self.dead_messages,
            "redelivered_after_visibility": list(
                self.redelivered_after_visibility),
        }
