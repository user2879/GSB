"""Explicit HTTP/2-like stream lifecycle transition table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import WorkbenchError


IDLE = "idle"
OPEN = "open"
HALF_CLOSED_LOCAL = "half_closed_local"
HALF_CLOSED_REMOTE = "half_closed_remote"
CLOSED = "closed"

STATES = (IDLE, OPEN, HALF_CLOSED_LOCAL, HALF_CLOSED_REMOTE, CLOSED)
ACTIVE_LOCAL_STATES = frozenset({OPEN, HALF_CLOSED_REMOTE})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    IDLE: frozenset({OPEN, CLOSED}),
    OPEN: frozenset({HALF_CLOSED_LOCAL, HALF_CLOSED_REMOTE, CLOSED}),
    HALF_CLOSED_LOCAL: frozenset({CLOSED}),
    HALF_CLOSED_REMOTE: frozenset({HALF_CLOSED_LOCAL, CLOSED}),
    CLOSED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class Transition:
    stream_id: int
    event_seq: int | None
    frame_seq: int | None
    from_state: str
    to_state: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "event_seq": self.event_seq,
            "frame_seq": self.frame_seq,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "Transition":
        return cls(
            stream_id=int(raw["stream_id"]),
            event_seq=None if raw.get("event_seq") is None else int(raw["event_seq"]),
            frame_seq=None if raw.get("frame_seq") is None else int(raw["frame_seq"]),
            from_state=str(raw["from_state"]),
            to_state=str(raw["to_state"]),
            reason=str(raw["reason"]),
        )


def check_transition(current: str, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


def transition(
    stream_id: int,
    current: str,
    target: str,
    *,
    event_seq: int | None,
    frame_seq: int | None,
    reason: str,
) -> Transition:
    if current not in ALLOWED_TRANSITIONS:
        raise WorkbenchError(
            "E_STATE",
            f"unknown stream state {current!r}",
            field="stream_state",
            event_seq=event_seq,
            frame_seq=frame_seq,
        )
    if not check_transition(current, target):
        raise WorkbenchError(
            "E_STATE",
            f"stream {stream_id} cannot transition {current} -> {target}",
            field="stream_state",
            event_seq=event_seq,
            frame_seq=frame_seq,
            details={
                "stream_id": stream_id,
                "from_state": current,
                "to_state": target,
                "reason": reason,
            },
        )
    return Transition(stream_id, event_seq, frame_seq, current, target, reason)
