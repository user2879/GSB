"""Domain models shared by validation, simulation, persistence, and UI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConnectionState(StrEnum):
    CLOSED = "CLOSED"
    SYN_SENT = "SYN_SENT"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT = "FIN_WAIT"
    CLOSE_WAIT = "CLOSE_WAIT"


class EventType(StrEnum):
    SYN = "SYN"
    SYN_ACK = "SYN_ACK"
    SEND = "SEND"
    ACK = "ACK"
    DUP_ACK = "DUP_ACK"
    TIMEOUT = "TIMEOUT"
    FIN = "FIN"
    FIN_ACK = "FIN_ACK"
    PEER_FIN = "PEER_FIN"
    CLOSE = "CLOSE"


STATE_EVENTS = frozenset(
    {
        EventType.SYN,
        EventType.SYN_ACK,
        EventType.FIN,
        EventType.FIN_ACK,
        EventType.PEER_FIN,
        EventType.CLOSE,
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetConfig:
    max_events: int
    max_sent_bytes: int
    max_elapsed_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_events": self.max_events,
            "max_sent_bytes": self.max_sent_bytes,
            "max_elapsed_ms": self.max_elapsed_ms,
        }


@dataclass(frozen=True)
class CongestionConfig:
    initial_cwnd_mss: int
    initial_ssthresh_bytes: int
    rtt_scale: int
    initial_rto_ms: int
    min_rto_ms: int
    max_rto_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "initial_cwnd_mss": self.initial_cwnd_mss,
            "initial_ssthresh_bytes": self.initial_ssthresh_bytes,
            "rtt_scale": self.rtt_scale,
            "initial_rto_ms": self.initial_rto_ms,
            "min_rto_ms": self.min_rto_ms,
            "max_rto_ms": self.max_rto_ms,
        }


@dataclass(frozen=True)
class SessionConfig:
    schema_version: int
    config_version: int
    session_id: str
    initial_seq: int
    window_limit: int
    mss: int
    peer_script: str
    budget: BudgetConfig
    congestion: CongestionConfig
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    digest: str = ""

    @property
    def first_data_seq(self) -> int:
        return self.initial_seq + 1

    @property
    def max_data_seq(self) -> int:
        return self.first_data_seq + self.budget.max_sent_bytes

    @property
    def initial_cwnd_bytes(self) -> int:
        return self.congestion.initial_cwnd_mss * self.mss

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.raw)
        return result


@dataclass(frozen=True)
class PeerResponse:
    for_seq: int
    action: str
    ack_seq: int | None = None
    delay_ms: int = 0
    reorder: bool = False


@dataclass(frozen=True)
class PeerScript:
    schema_version: int
    script_version: int
    default_policy: str
    default_delay_ms: int
    responses: tuple[PeerResponse, ...]
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    digest: str = ""
    response_map: dict[int, PeerResponse] = field(
        default_factory=dict, compare=False, repr=False
    )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class Event:
    event_type: str
    recorded_mono_ms: int
    seq: int | None = None
    ack_seq: int | None = None
    payload_bytes: int = 0
    retransmitted: bool = False
    rtt_sample_ms: int | None = None
    event_seq: int = 0
    note: str = ""

    @property
    def packet_type(self) -> str:
        return self.event_type

    def with_seq(self, event_seq: int) -> Event:
        return Event(
            event_type=self.event_type,
            recorded_mono_ms=self.recorded_mono_ms,
            seq=self.seq,
            ack_seq=self.ack_seq,
            payload_bytes=self.payload_bytes,
            retransmitted=self.retransmitted,
            rtt_sample_ms=self.rtt_sample_ms,
            event_seq=event_seq,
            note=self.note,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "event_type": self.event_type,
            "recorded_mono_ms": self.recorded_mono_ms,
            "seq": self.seq,
            "ack_seq": self.ack_seq,
            "payload_bytes": self.payload_bytes,
            "retransmitted": self.retransmitted,
            "rtt_sample_ms": self.rtt_sample_ms,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Event:
        return cls(
            event_type=str(value.get("event_type", "")),
            recorded_mono_ms=int(value.get("recorded_mono_ms", 0)),
            seq=None if value.get("seq") is None else int(value["seq"]),
            ack_seq=None if value.get("ack_seq") is None else int(value["ack_seq"]),
            payload_bytes=int(value.get("payload_bytes", 0)),
            retransmitted=bool(value.get("retransmitted", False)),
            rtt_sample_ms=None
            if value.get("rtt_sample_ms") is None
            else int(value["rtt_sample_ms"]),
            event_seq=int(value.get("event_seq", 0)),
            note=str(value.get("note", "")),
        )


@dataclass(frozen=True)
class ItemResult:
    key: str
    value: Any
    revision: int
    basis_event_seq: int
    depends_on_items: tuple[str, ...]
    depends_on_events: tuple[int, ...]
    config_version: int
    script_version: int
    stale: bool = False
    last_affected_event_seq: int = 0

    @property
    def value_json(self) -> str:
        return canonical_json(self.value)

    @property
    def value_digest(self) -> str:
        return hashlib.sha256(self.value_json.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "revision": self.revision,
            "basis_event_seq": self.basis_event_seq,
            "depends_on_items": list(self.depends_on_items),
            "depends_on_events": list(self.depends_on_events),
            "config_version": self.config_version,
            "script_version": self.script_version,
            "stale": self.stale,
            "last_affected_event_seq": self.last_affected_event_seq,
        }

    def identity(self) -> tuple[Any, ...]:
        return (
            self.key,
            self.value_json,
            self.revision,
            self.basis_event_seq,
            self.depends_on_items,
            self.depends_on_events,
            self.config_version,
            self.script_version,
            self.stale,
            self.last_affected_event_seq,
        )


@dataclass(frozen=True)
class FieldChange:
    field: str
    old_value: Any
    new_value: Any

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "old": self.old_value, "new": self.new_value}


@dataclass(frozen=True)
class ReplacementReport:
    reason: str
    affected_fields: tuple[FieldChange, ...]
    old_config_version: int | None
    old_script_version: int | None
    new_config_version: int
    new_script_version: int
    archived_fields: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "affected_fields": [item.as_dict() for item in self.affected_fields],
            "old_config_version": self.old_config_version,
            "old_script_version": self.old_script_version,
            "new_config_version": self.new_config_version,
            "new_script_version": self.new_script_version,
            "archived_fields": self.archived_fields,
        }


def flatten_values(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten_values(
                f"{prefix}.{key}" if prefix else str(key), value[key]
            )
    elif isinstance(value, list):
        yield prefix, value
    else:
        yield prefix, value
