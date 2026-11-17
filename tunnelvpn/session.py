"""Mutable runtime session state persisted by snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import QueueEvent


@dataclass
class SessionState:
    session_id: str
    route_id: str
    peer_id: str
    sequence: int
    start_ms: int
    status: str = "QUEUED"
    processing_started_ms: int | None = None
    end_ms: int | None = None
    payload_bytes: int = 0
    forwarded_bytes: int = 0
    wire_bytes: int = 0
    queue_events: list[QueueEvent] = field(default_factory=list)
    business_code: str = "PENDING"
    encapsulation_digest: str = ""
    peer_payload_digest: str | None = None
    fragment_count: int = 0
    payload_b64: str | None = None
    error: dict[str, Any] | None = None
    replay_of: str | None = None
    requires_rerun: bool = False
    _queue_start_ms: int | None = field(default=None, repr=False)

    @property
    def queue_count(self) -> int:
        return len(self.queue_events)

    @property
    def total_queue_ms(self) -> int:
        return sum(event.duration_ms for event in self.queue_events)

    def begin_queue(self, now_ms: int) -> None:
        if self._queue_start_ms is None:
            self._queue_start_ms = now_ms

    def end_queue(self, now_ms: int) -> None:
        if self._queue_start_ms is None:
            return
        start = self._queue_start_ms
        self._queue_start_ms = None
        self.queue_events.append(QueueEvent(start, max(start, now_ms)))

    def finish_queue(self, now_ms: int) -> None:
        self.end_queue(now_ms)

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "route_id": self.route_id,
            "peer_id": self.peer_id,
            "sequence": self.sequence,
            "start_ms": self.start_ms,
            "processing_started_ms": self.processing_started_ms,
            "end_ms": self.end_ms,
            "payload_bytes": self.payload_bytes,
            "forwarded_bytes": self.forwarded_bytes,
            "wire_bytes": self.wire_bytes,
            "queue_events": [event.to_dict() for event in self.queue_events],
            "queue_count": self.queue_count,
            "total_queue_ms": self.total_queue_ms,
            "status": self.status,
            "business_code": self.business_code,
            "encapsulation_digest": self.encapsulation_digest,
            "peer_payload_digest": self.peer_payload_digest,
            "fragment_count": self.fragment_count,
            "error": self.error,
            "replay_of": self.replay_of,
            "requires_rerun": self.requires_rerun,
        }
        if include_payload:
            result["payload_b64"] = self.payload_b64
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionState":
        state = cls(
            session_id=str(raw["session_id"]),
            route_id=str(raw["route_id"]),
            peer_id=str(raw["peer_id"]),
            sequence=int(raw["sequence"]),
            start_ms=int(raw["start_ms"]),
            status=str(raw.get("status", "UNKNOWN")),
            processing_started_ms=raw.get("processing_started_ms"),
            end_ms=raw.get("end_ms"),
            payload_bytes=int(raw.get("payload_bytes", 0)),
            forwarded_bytes=int(raw.get("forwarded_bytes", 0)),
            wire_bytes=int(raw.get("wire_bytes", 0)),
            queue_events=[
                QueueEvent.from_dict(item) for item in raw.get("queue_events", [])
            ],
            business_code=str(raw.get("business_code", "PENDING")),
            encapsulation_digest=str(raw.get("encapsulation_digest", "")),
            peer_payload_digest=raw.get("peer_payload_digest"),
            fragment_count=int(raw.get("fragment_count", 0)),
            payload_b64=raw.get("payload_b64"),
            error=raw.get("error"),
            replay_of=raw.get("replay_of"),
            requires_rerun=bool(raw.get("requires_rerun", False)),
        )
        return state
