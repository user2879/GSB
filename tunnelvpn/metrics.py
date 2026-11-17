"""Single-write session ledger and integer fixed-point statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import ErrorCode, TunnelError

STATS_SCALE = 1000


def round_half_up(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator * 2 + denominator) // (2 * denominator)
    return -((-numerator * 2 + denominator) // (2 * denominator))


@dataclass(frozen=True)
class QueueEvent:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, int]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueueEvent":
        return cls(start_ms=int(raw["start_ms"]), end_ms=int(raw["end_ms"]))


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    route_id: str
    peer_id: str
    sequence: int
    start_ms: int
    end_ms: int
    payload_bytes: int
    forwarded_bytes: int
    wire_bytes: int
    queue_events: tuple[QueueEvent, ...]
    status: str
    business_code: str
    encapsulation_digest: str
    fragment_count: int
    peer_payload_digest: str | None = None
    replay_of: str | None = None
    error_detail: dict[str, Any] | None = None

    @property
    def queue_count(self) -> int:
        return len(self.queue_events)

    @property
    def total_queue_ms(self) -> int:
        return sum(event.duration_ms for event in self.queue_events)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "route_id": self.route_id,
            "peer_id": self.peer_id,
            "sequence": self.sequence,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "payload_bytes": self.payload_bytes,
            "forwarded_bytes": self.forwarded_bytes,
            "wire_bytes": self.wire_bytes,
            "queue_count": self.queue_count,
            "queue_events": [event.to_dict() for event in self.queue_events],
            "total_queue_ms": self.total_queue_ms,
            "status": self.status,
            "business_code": self.business_code,
            "encapsulation_digest": self.encapsulation_digest,
            "fragment_count": self.fragment_count,
        }
        if self.peer_payload_digest is not None:
            result["peer_payload_digest"] = self.peer_payload_digest
        if self.replay_of is not None:
            result["replay_of"] = self.replay_of
        if self.error_detail is not None:
            result["error_detail"] = self.error_detail
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=str(raw["session_id"]),
            route_id=str(raw["route_id"]),
            peer_id=str(raw["peer_id"]),
            sequence=int(raw["sequence"]),
            start_ms=int(raw["start_ms"]),
            end_ms=int(raw["end_ms"]),
            payload_bytes=int(raw["payload_bytes"]),
            forwarded_bytes=int(raw.get("forwarded_bytes", raw["payload_bytes"])),
            wire_bytes=int(raw["wire_bytes"]),
            queue_events=tuple(
                QueueEvent.from_dict(item) for item in raw.get("queue_events", [])
            ),
            status=str(raw["status"]),
            business_code=str(raw["business_code"]),
            encapsulation_digest=str(raw["encapsulation_digest"]),
            fragment_count=int(raw.get("fragment_count", 1)),
            peer_payload_digest=raw.get("peer_payload_digest"),
            replay_of=raw.get("replay_of"),
            error_detail=raw.get("error_detail"),
        )


class Ledger:
    """Ordered session ledger with strict exactly-one-record semantics."""

    def __init__(self, records: Iterable[SessionRecord] = ()) -> None:
        self._records: dict[str, SessionRecord] = {}
        for record in records:
            self.record(record)

    def record(self, entry: SessionRecord) -> None:
        if not entry.session_id:
            raise TunnelError(ErrorCode.E_STATE, "会话标识不能为空")
        if entry.session_id in self._records:
            raise TunnelError(
                ErrorCode.E_STATE,
                "同一会话标识的账本记录只能写入一次",
                tunnel_id=entry.route_id,
                session_id=entry.session_id,
            )
        if entry.start_ms < 0 or entry.end_ms < entry.start_ms:
            raise TunnelError(
                ErrorCode.E_STATE,
                "会话时间区间非法",
                tunnel_id=entry.route_id,
                session_id=entry.session_id,
            )
        if entry.payload_bytes < 0 or entry.forwarded_bytes < 0 or entry.wire_bytes < 0:
            raise TunnelError(
                ErrorCode.E_STATE,
                "会话字节数不能为负",
                tunnel_id=entry.route_id,
                session_id=entry.session_id,
            )
        self._records[entry.session_id] = entry

    def contains(self, session_id: str) -> bool:
        return session_id in self._records

    def get(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    def records(self) -> tuple[SessionRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda entry: (entry.start_ms, entry.session_id),
            )
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.records()]

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {"records": self.to_dicts()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dicts(cls, records: Iterable[dict[str, Any]]) -> "Ledger":
        return cls(SessionRecord.from_dict(item) for item in records)

    def statistics(self) -> dict[str, Any]:
        groups: dict[str | None, list[SessionRecord]] = {None: list(self.records())}
        route_ids = sorted({entry.route_id for entry in self._records.values()})
        for route_id in route_ids:
            groups[route_id] = [
                entry
                for entry in self._records.values()
                if entry.route_id == route_id
            ]
        result: dict[str, Any] = {
            "scale": STATS_SCALE,
            "units": {
                "throughput": "milli-KB/s (KB/s * 1000)",
                "average_queue_duration": "milli-ms (ms * 1000)",
            },
            "quantization_error_max": {
                "throughput_kbps": 0.5 / STATS_SCALE,
                "average_queue_duration_ms": 0.5 / STATS_SCALE,
            },
            "global": _aggregate(groups[None]),
            "routes": {
                route_id: _aggregate(groups[route_id]) for route_id in route_ids
            },
            "business_code_counts": _code_counts(self._records.values()),
            "ledger_digest": self.digest(),
        }
        return result


def _aggregate(records: list[SessionRecord]) -> dict[str, Any]:
    count = len(records)
    payload_bytes = sum(entry.payload_bytes for entry in records)
    forwarded_bytes = sum(entry.forwarded_bytes for entry in records)
    wire_bytes = sum(entry.wire_bytes for entry in records)
    queue_ms = sum(entry.total_queue_ms for entry in records)
    queue_count = sum(entry.queue_count for entry in records)
    if count:
        span_start = min(entry.start_ms for entry in records)
        span_end = max(entry.end_ms for entry in records)
        span_ms = span_end - span_start
    else:
        span_start = span_end = span_ms = 0
    throughput_millikbps = (
        round_half_up(forwarded_bytes * 1_000_000, 1024 * span_ms) if span_ms > 0 else 0
    )
    average_queue_fixed = (
        round_half_up(queue_ms * STATS_SCALE, count) if count else 0
    )
    return {
        "session_count": count,
        "completed_count": sum(entry.status == "COMPLETED" for entry in records),
        "failed_count": sum(entry.status == "FAILED" for entry in records),
        "payload_bytes": payload_bytes,
        "forwarded_bytes": forwarded_bytes,
        "wire_bytes": wire_bytes,
        "span_start_ms": span_start,
        "span_end_ms": span_end,
        "span_ms": span_ms,
        "queue_count": queue_count,
        "total_queue_ms": queue_ms,
        "throughput_kbps_fixed": throughput_millikbps,
        "average_queue_ms_fixed": average_queue_fixed,
        "throughput_kbps": f"{throughput_millikbps // STATS_SCALE}.{throughput_millikbps % STATS_SCALE:03d}",
        "average_queue_duration_ms": f"{average_queue_fixed // STATS_SCALE}.{average_queue_fixed % STATS_SCALE:03d}",
    }


def _code_counts(records: Iterable[SessionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.business_code] = counts.get(record.business_code, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def compare_statistics(
    expected: dict[str, Any], actual: dict[str, Any], *, prefix: str = "statistics"
) -> tuple[str, Any, Any] | None:
    """Return the first deterministic statistic mismatch, never ``ledger_digest``."""

    if expected.get("scale") != actual.get("scale"):
        return f"{prefix}.scale", expected.get("scale"), actual.get("scale")
    groups = ["global", *sorted(set(expected.get("routes", {})) | set(actual.get("routes", {})))]
    for group in groups:
        expected_group = expected.get("global", {}) if group == "global" else expected.get("routes", {}).get(group, {})
        actual_group = actual.get("global", {}) if group == "global" else actual.get("routes", {}).get(group, {})
        fields = [
            "session_count",
            "completed_count",
            "failed_count",
            "payload_bytes",
            "forwarded_bytes",
            "wire_bytes",
            "span_start_ms",
            "span_end_ms",
            "span_ms",
            "queue_count",
            "total_queue_ms",
            "throughput_kbps_fixed",
            "average_queue_ms_fixed",
        ]
        for field in fields:
            if expected_group.get(field) != actual_group.get(field):
                return (
                    f"{prefix}.{group}.{field}",
                    expected_group.get(field),
                    actual_group.get(field),
                )
    if expected.get("business_code_counts") != actual.get("business_code_counts"):
        return (
            f"{prefix}.business_code_counts",
            expected.get("business_code_counts"),
            actual.get("business_code_counts"),
        )
    return None
