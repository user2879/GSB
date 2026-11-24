"""Session declaration parsing, strong validation, and content digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import WorkbenchError


MAX_WINDOW = (1 << 31) - 1
MAX_STREAM_ID = (1 << 31) - 1
MAX_ERROR_CODE = (1 << 32) - 1
SUPPORTED_FRAME_SIZES = frozenset({16_384, 32_768, 65_536})
DEFAULT_STREAM_WINDOW = 65_535
STRATEGIES = frozenset({"round_robin", "proportional"})


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _spec_error(field: str, message: str, event_seq: int | None = None) -> WorkbenchError:
    return WorkbenchError(
        code="E_SPEC",
        message=message,
        field=field,
        event_seq=event_seq,
    )


def _as_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    event_seq: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _spec_error(field, "must be an integer", event_seq)
    if minimum is not None and value < minimum:
        raise _spec_error(field, f"must be >= {minimum}", event_seq)
    if maximum is not None and value > maximum:
        raise _spec_error(field, f"must be <= {maximum}", event_seq)
    return value


def _as_str(value: Any, field: str, *, event_seq: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise _spec_error(field, "must be a non-empty string", event_seq)
    return value


def _as_object(value: Any, field: str, *, event_seq: int | None = None) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _spec_error(field, "must be an object", event_seq)
    return value


def _reject_unknown(
    obj: Mapping[str, Any],
    allowed: Iterable[str],
    prefix: str,
    *,
    event_seq: int | None = None,
) -> None:
    allowed_set = frozenset(allowed)
    for key in obj:
        if key not in allowed_set:
            raise _spec_error(f"{prefix}.{key}", "unknown field", event_seq)


@dataclass(frozen=True, slots=True)
class StreamSpec:
    stream_id: int
    priority: int
    request_body_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "stream_id": self.stream_id,
            "priority": self.priority,
            "request_body_bytes": self.request_body_bytes,
        }


@dataclass(frozen=True, slots=True)
class EventSpec:
    seq: int
    arrival_ms: int
    kind: str
    scope: str | None = None
    stream_id: int | None = None
    increment: int | None = None
    priority: int | None = None
    strategy: str | None = None
    error_code: int = 0
    last_stream_id: int | None = None
    source_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "arrival_ms": self.arrival_ms,
            "type": self.kind,
        }
        if self.kind == "window_update":
            result["scope"] = self.scope
            if self.scope == "stream":
                result["stream_id"] = self.stream_id
            result["increment"] = self.increment
        elif self.kind == "rst_stream":
            result["stream_id"] = self.stream_id
            result["error_code"] = self.error_code
        elif self.kind == "goaway":
            result["last_stream_id"] = self.last_stream_id
            result["error_code"] = self.error_code
        elif self.kind == "priority_change":
            result["stream_id"] = self.stream_id
            result["priority"] = self.priority
        elif self.kind == "strategy_change":
            result["strategy"] = self.strategy
        return result


@dataclass(frozen=True, slots=True)
class ConnectionSpec:
    initial_window: int
    initial_stream_window: int
    max_concurrent_streams: int
    max_frame_size: int
    scheduling_quantum: int

    def to_dict(self) -> dict[str, int]:
        return {
            "initial_window": self.initial_window,
            "initial_stream_window": self.initial_stream_window,
            "max_concurrent_streams": self.max_concurrent_streams,
            "max_frame_size": self.max_frame_size,
            "scheduling_quantum": self.scheduling_quantum,
        }


@dataclass(frozen=True, slots=True)
class SessionSpec:
    format_version: int
    session_id: str
    declared_version: int
    connection: ConnectionSpec
    streams: tuple[StreamSpec, ...]
    events: tuple[EventSpec, ...]
    event_offset: int
    content_digest: str

    @property
    def stream_ids(self) -> tuple[int, ...]:
        return tuple(stream.stream_id for stream in self.streams)

    @property
    def stream_index(self) -> dict[int, StreamSpec]:
        return {stream.stream_id: stream for stream in self.streams}

    @property
    def next_event_seq(self) -> int:
        return self.event_offset + len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "session_id": self.session_id,
            "declared_version": self.declared_version,
            "connection": self.connection.to_dict(),
            "streams": [stream.to_dict() for stream in self.streams],
            "events": [event.to_dict() for event in self.events],
        }


_TOP_LEVEL_KEYS = {
    "format_version",
    "session_id",
    "declared_version",
    "connection",
    "connection_initial_window",
    "initial_window",
    "initial_stream_window",
    "max_concurrent_streams",
    "max_frame_size",
    "scheduling_quantum",
    "streams",
    "events",
    "notes",
}
_CONNECTION_KEYS = {
    "initial_window",
    "initial_window_size",
    "initial_stream_window",
    "max_concurrent_streams",
    "max_frame_size",
    "scheduling_quantum",
}
_STREAM_KEYS = {"stream_id", "priority", "request_body_bytes"}
_EVENT_KEYS = {
    "seq",
    "arrival_ms",
    "type",
    "scope",
    "stream_id",
    "increment",
    "delta",
    "window_increment",
    "priority",
    "weight",
    "strategy",
    "policy",
    "error_code",
    "last_stream_id",
    "comment",
}


def _parse_event(raw: Any, index: int, event_offset: int, stream_ids: set[int]) -> EventSpec:
    obj = _as_object(raw, f"events[{index}]")
    raw_seq = obj.get("seq")
    if isinstance(raw_seq, bool) or not isinstance(raw_seq, int):
        raise _spec_error(f"events[{index}].seq", "must be an integer")
    seq = raw_seq
    _reject_unknown(obj, _EVENT_KEYS, f"events[{index}]", event_seq=seq)
    if index == 0:
        if seq not in (0, 1):
            raise _spec_error(f"events[{index}].seq", "sequence must start at 0 or 1", seq)
        event_offset = seq
    elif seq != event_offset + index:
        raise _spec_error(
            f"events[{index}].seq",
            f"expected continuous sequence value {event_offset + index}",
            seq,
        )

    arrival_ms = _as_int(
        obj.get("arrival_ms"),
        f"events[{index}].arrival_ms",
        minimum=0,
        event_seq=seq,
    )
    raw_type = _as_str(obj.get("type"), f"events[{index}].type", event_seq=seq)
    normalized = raw_type.strip().lower().replace("-", "_")
    type_alias = {
        "window_update": "window_update",
        "connection_window_update": "window_update",
        "stream_window_update": "window_update",
        "rst_stream": "rst_stream",
        "reset_stream": "rst_stream",
        "goaway": "goaway",
        "priority_change": "priority_change",
        "weight_change": "priority_change",
        "strategy_change": "strategy_change",
        "strategy_switch": "strategy_change",
    }
    if normalized not in type_alias:
        raise _spec_error(f"events[{index}].type", f"unsupported event type {raw_type!r}", seq)
    kind = type_alias[normalized]

    if kind == "window_update":
        scope_value = obj.get("scope")
        stream_id_value = obj.get("stream_id")
        if normalized == "connection_window_update" or scope_value == "connection":
            scope = "connection"
        elif normalized == "stream_window_update" or scope_value == "stream":
            scope = "stream"
        elif stream_id_value in (None, 0):
            scope = "connection"
        else:
            scope = "stream"
        if scope_value is not None and scope_value not in {"connection", "stream"}:
            raise _spec_error(f"events[{index}].scope", "must be 'connection' or 'stream'", seq)
        increment = obj.get("increment", obj.get("delta", obj.get("window_increment")))
        increment = _as_int(
            increment,
            f"events[{index}].increment",
            minimum=0,
            maximum=MAX_WINDOW,
            event_seq=seq,
        )
        stream_id = None
        if scope == "stream":
            stream_id = _as_int(
                stream_id_value,
                f"events[{index}].stream_id",
                minimum=1,
                maximum=MAX_STREAM_ID,
                event_seq=seq,
            )
            if stream_id not in stream_ids:
                raise _spec_error(
                    f"events[{index}].stream_id",
                    f"unknown stream {stream_id}",
                    seq,
                )
        return EventSpec(
            seq=seq,
            arrival_ms=arrival_ms,
            kind=kind,
            scope=scope,
            stream_id=stream_id,
            increment=increment,
            source_type=raw_type,
        )

    if kind == "rst_stream":
        stream_id = _as_int(
            obj.get("stream_id"),
            f"events[{index}].stream_id",
            minimum=1,
            maximum=MAX_STREAM_ID,
            event_seq=seq,
        )
        if stream_id not in stream_ids:
            raise _spec_error(f"events[{index}].stream_id", f"unknown stream {stream_id}", seq)
        error_code = _as_int(
            obj.get("error_code", 0),
            f"events[{index}].error_code",
            minimum=0,
            maximum=MAX_ERROR_CODE,
            event_seq=seq,
        )
        return EventSpec(
            seq=seq,
            arrival_ms=arrival_ms,
            kind=kind,
            stream_id=stream_id,
            error_code=error_code,
            source_type=raw_type,
        )

    if kind == "goaway":
        last_stream_id = _as_int(
            obj.get("last_stream_id"),
            f"events[{index}].last_stream_id",
            minimum=0,
            maximum=MAX_STREAM_ID,
            event_seq=seq,
        )
        error_code = _as_int(
            obj.get("error_code", 0),
            f"events[{index}].error_code",
            minimum=0,
            maximum=MAX_ERROR_CODE,
            event_seq=seq,
        )
        return EventSpec(
            seq=seq,
            arrival_ms=arrival_ms,
            kind=kind,
            last_stream_id=last_stream_id,
            error_code=error_code,
            source_type=raw_type,
        )

    if kind == "priority_change":
        stream_id = _as_int(
            obj.get("stream_id"),
            f"events[{index}].stream_id",
            minimum=1,
            maximum=MAX_STREAM_ID,
            event_seq=seq,
        )
        if stream_id not in stream_ids:
            raise _spec_error(f"events[{index}].stream_id", f"unknown stream {stream_id}", seq)
        priority = _as_int(
            obj.get("priority", obj.get("weight")),
            f"events[{index}].priority",
            minimum=1,
            maximum=256,
            event_seq=seq,
        )
        return EventSpec(
            seq=seq,
            arrival_ms=arrival_ms,
            kind=kind,
            stream_id=stream_id,
            priority=priority,
            source_type=raw_type,
        )

    strategy_value = obj.get("strategy", obj.get("policy"))
    strategy = _as_str(strategy_value, f"events[{index}].strategy", event_seq=seq)
    strategy = strategy.strip().lower().replace("-", "_")
    if strategy not in STRATEGIES:
        raise _spec_error(
            f"events[{index}].strategy",
            "must be 'round_robin' or 'proportional'",
            seq,
        )
    return EventSpec(
        seq=seq,
        arrival_ms=arrival_ms,
        kind=kind,
        strategy=strategy,
        source_type=raw_type,
    )


def parse_session(raw: Any) -> SessionSpec:
    obj = _as_object(raw, "$")
    _reject_unknown(obj, _TOP_LEVEL_KEYS, "$")

    format_version = _as_int(obj.get("format_version", 1), "format_version", minimum=1)
    if format_version != 1:
        raise _spec_error("format_version", "only integer format version 1 is supported")
    session_id = _as_str(obj.get("session_id"), "session_id")
    declared_version = _as_int(
        obj.get("declared_version", 1),
        "declared_version",
        minimum=1,
    )

    connection_obj = _as_object(obj.get("connection", {}), "connection")
    _reject_unknown(connection_obj, _CONNECTION_KEYS, "connection")

    raw_initial = obj.get(
        "connection_initial_window",
        obj.get("initial_window", connection_obj.get("initial_window", 65_535)),
    )
    initial_window = _as_int(
        raw_initial,
        "connection.initial_window",
        minimum=0,
        maximum=MAX_WINDOW,
    )
    initial_stream_window = _as_int(
        obj.get(
            "initial_stream_window",
            connection_obj.get("initial_stream_window", DEFAULT_STREAM_WINDOW),
        ),
        "connection.initial_stream_window",
        minimum=0,
        maximum=MAX_WINDOW,
    )
    max_concurrent_streams = _as_int(
        obj.get(
            "max_concurrent_streams",
            connection_obj.get("max_concurrent_streams", 100),
        ),
        "connection.max_concurrent_streams",
        minimum=1,
        maximum=65_535,
    )
    max_frame_size = _as_int(
        obj.get("max_frame_size", connection_obj.get("max_frame_size", 16_384)),
        "connection.max_frame_size",
    )
    if max_frame_size not in SUPPORTED_FRAME_SIZES:
        raise _spec_error(
            "connection.max_frame_size",
            f"supported values are {sorted(SUPPORTED_FRAME_SIZES)}",
        )

    stream_hint = obj.get("streams", [])
    stream_count_hint = len(stream_hint) if isinstance(stream_hint, list) else 0
    default_quantum = 12 if stream_count_hint >= 512 else 5 if stream_count_hint >= 16 else 3
    scheduling_quantum = _as_int(
        obj.get("scheduling_quantum", connection_obj.get("scheduling_quantum", default_quantum)),
        "connection.scheduling_quantum",
        minimum=1,
        maximum=4096,
    )

    streams_raw = obj.get("streams")
    if not isinstance(streams_raw, list):
        raise _spec_error("streams", "must be an array")
    streams: list[StreamSpec] = []
    seen_ids: set[int] = set()
    for index, raw_stream in enumerate(streams_raw):
        stream_obj = _as_object(raw_stream, f"streams[{index}]")
        _reject_unknown(stream_obj, _STREAM_KEYS, f"streams[{index}]")
        stream_id = _as_int(
            stream_obj.get("stream_id"),
            f"streams[{index}].stream_id",
            minimum=1,
            maximum=MAX_STREAM_ID,
        )
        if stream_id % 2 == 0:
            raise _spec_error(f"streams[{index}].stream_id", "must be odd")
        if stream_id in seen_ids:
            raise _spec_error(f"streams[{index}].stream_id", "must be unique")
        priority = _as_int(
            stream_obj.get("priority"),
            f"streams[{index}].priority",
            minimum=1,
            maximum=256,
        )
        body_bytes = _as_int(
            stream_obj.get("request_body_bytes"),
            f"streams[{index}].request_body_bytes",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        seen_ids.add(stream_id)
        streams.append(StreamSpec(stream_id, priority, body_bytes))

    if not streams:
        raise _spec_error("streams", "must contain at least one stream")

    events_raw = obj.get("events")
    if not isinstance(events_raw, list):
        raise _spec_error("events", "must be an array")
    events: list[EventSpec] = []
    event_offset = events_raw[0].get("seq", 0) if events_raw and isinstance(events_raw[0], dict) else 0
    if isinstance(event_offset, bool) or not isinstance(event_offset, int) or event_offset not in (0, 1):
        event_offset = 0
    previous_arrival = -1
    for index, raw_event in enumerate(events_raw):
        event = _parse_event(raw_event, index, event_offset, seen_ids)
        if event.arrival_ms < previous_arrival:
            raise _spec_error(
                f"events[{index}].arrival_ms",
                "must be monotonically non-decreasing",
                event.seq,
            )
        previous_arrival = event.arrival_ms
        events.append(event)
    if events:
        event_offset = events[0].seq

    normalized_payload = {
        "format_version": format_version,
        "session_id": session_id,
        "declared_version": declared_version,
        "connection": ConnectionSpec(
            initial_window=initial_window,
            initial_stream_window=initial_stream_window,
            max_concurrent_streams=max_concurrent_streams,
            max_frame_size=max_frame_size,
            scheduling_quantum=scheduling_quantum,
        ).to_dict(),
        "streams": [stream.to_dict() for stream in streams],
        "events": [event.to_dict() for event in events],
    }
    connection_spec = ConnectionSpec(
        initial_window=initial_window,
        initial_stream_window=initial_stream_window,
        max_concurrent_streams=max_concurrent_streams,
        max_frame_size=max_frame_size,
        scheduling_quantum=scheduling_quantum,
    )
    return SessionSpec(
        format_version=format_version,
        session_id=session_id,
        declared_version=declared_version,
        connection=connection_spec,
        streams=tuple(streams),
        events=tuple(events),
        event_offset=event_offset,
        content_digest=digest_value(normalized_payload),
    )


def load_session(path: str | Path) -> SessionSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError("E_IO", f"cannot read session file: {exc}", field="path") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkbenchError(
            "E_SPEC",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            field="$",
        ) from exc
    return parse_session(raw)
