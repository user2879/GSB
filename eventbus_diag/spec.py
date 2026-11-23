"""Strict loading and validation for declarations and NDJSON events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest_value
from .errors import SpecError
from .models import Channel, Declaration, Event, Subscriber


_KEY_RE = re.compile(r'"((?:[^"\\]|\\.)+)"\s*:')


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class SourceIndex:
    """Small, dependency-free map from JSON keys to their source lines.

    This is intentionally not a second JSON parser.  It only records lines for
    quoted object keys.  Semantic errors for repeated arrays use occurrence
    numbers, which is sufficient for the fixed declaration schema.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.occurrences: dict[str, list[int]] = {}
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in _KEY_RE.finditer(line):
                key = match.group(1)
                self.occurrences.setdefault(key, []).append(line_no)

    def line_for(self, key: str, occurrence: int = 0, fallback: int = 1) -> int:
        lines = self.occurrences.get(key, [])
        if not lines:
            return fallback
        if occurrence < 0:
            occurrence = 0
        if occurrence >= len(lines):
            occurrence = len(lines) - 1
        return lines[occurrence]


@dataclass(frozen=True)
class Dataset:
    declaration: Declaration
    events: tuple[Event, ...]
    spec_path: str
    events_path: str
    spec_text: str
    events_digest: str
    content_digest: str


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise SpecError(
            f"file does not exist: {path}", field="$file", line=None
        ) from exc
    except OSError as exc:
        raise SpecError(
            f"cannot read file {path}: {exc}", field="$file", line=None
        ) from exc


def _json_error(exc: json.JSONDecodeError) -> SpecError:
    return SpecError(
        f"invalid JSON: {exc.msg}",
        field="$json",
        line=int(exc.lineno),
        details={"column": int(exc.colno)},
    )


def _require_object(value: Any, field: str, line: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{field} must be an object", field=field, line=line)
    return value


def _require_list(value: Any, field: str, line: int) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(f"{field} must be an array", field=field, line=line)
    return value


def _require_key(obj: dict[str, Any], key: str, field: str, line: int) -> Any:
    if key not in obj:
        raise SpecError(f"missing required field {field}", field=field, line=line)
    return obj[key]


def _require_string(value: Any, field: str, line: int, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SpecError(f"{field} must be a string", field=field, line=line)
    if nonempty and not value.strip():
        raise SpecError(f"{field} must not be empty", field=field, line=line)
    return value


def _require_positive_int(value: Any, field: str, line: int) -> int:
    if not _is_int(value) or value <= 0:
        raise SpecError(f"{field} must be a positive integer", field=field, line=line)
    return int(value)


def _require_int(value: Any, field: str, line: int) -> int:
    if not _is_int(value):
        raise SpecError(f"{field} must be an integer", field=field, line=line)
    return int(value)


def validate_declaration_object(
    obj: Any,
    *,
    source_text: str,
    source_path: str,
) -> Declaration:
    lines = SourceIndex(source_text)
    root = _require_object(obj, "$", 1)

    if "declaration_version" not in root:
        raise SpecError(
            "missing required field declaration_version",
            field="declaration_version",
            line=lines.line_for("subscribers", fallback=1),
        )
    declaration_version = _require_positive_int(
        root["declaration_version"],
        "declaration_version",
        lines.line_for("declaration_version"),
    )

    if "global_lease_limit" not in root:
        raise SpecError(
            "missing required field global_lease_limit",
            field="global_lease_limit",
            line=lines.line_for("subscribers", fallback=1),
        )
    global_lease_limit = _require_positive_int(
        root["global_lease_limit"],
        "global_lease_limit",
        lines.line_for("global_lease_limit"),
    )

    handshake_timeout_ms = 3000
    if "handshake_timeout_ms" in root:
        handshake_timeout_ms = _require_positive_int(
            root["handshake_timeout_ms"],
            "handshake_timeout_ms",
            lines.line_for("handshake_timeout_ms"),
        )

    raw_subscribers = _require_list(
        _require_key(root, "subscribers", "subscribers", lines.line_for("subscribers")),
        "subscribers",
        lines.line_for("subscribers"),
    )
    if not raw_subscribers:
        raise SpecError(
            "subscribers must contain at least one item",
            field="subscribers",
            line=lines.line_for("subscribers"),
        )

    if "channels" not in root:
        raise SpecError(
            "missing required field channels",
            field="channels",
            line=lines.line_for("subscribers", fallback=1),
        )
    raw_channels = _require_list(
        root["channels"], "channels", lines.line_for("channels")
    )
    if not raw_channels:
        raise SpecError(
            "channels must contain at least one item",
            field="channels",
            line=lines.line_for("channels"),
        )

    subscribers: list[Subscriber] = []
    subscriber_ids: set[str] = set()
    subscriber_ports: dict[int, str] = {}
    for index, raw_item in enumerate(raw_subscribers):
        item_line = lines.line_for("subscriber_id", index, lines.line_for("subscribers"))
        item = _require_object(raw_item, f"subscribers[{index}]", item_line)
        field_prefix = f"subscribers[{index}]"

        subscriber_id = _require_string(
            _require_key(item, "subscriber_id", f"{field_prefix}.subscriber_id", item_line),
            f"{field_prefix}.subscriber_id",
            lines.line_for("subscriber_id", index, item_line),
        )
        if subscriber_id in subscriber_ids:
            raise SpecError(
                f"duplicate subscriber_id {subscriber_id!r}",
                field=f"{field_prefix}.subscriber_id",
                line=lines.line_for("subscriber_id", index, item_line),
            )

        host = _require_string(
            _require_key(item, "host", f"{field_prefix}.host", item_line),
            f"{field_prefix}.host",
            lines.line_for("host", index, item_line),
        )
        if host != "127.0.0.1":
            raise SpecError(
                "subscriber host must be the literal loopback address 127.0.0.1",
                field=f"{field_prefix}.host",
                line=lines.line_for("host", index, item_line),
            )

        port = _require_positive_int(
            _require_key(item, "port", f"{field_prefix}.port", item_line),
            f"{field_prefix}.port",
            lines.line_for("port", index, item_line),
        )
        if port > 65535:
            raise SpecError(
                f"{field_prefix}.port must be between 1 and 65535",
                field=f"{field_prefix}.port",
                line=lines.line_for("port", index, item_line),
            )
        if port in subscriber_ports:
            raise SpecError(
                f"duplicate loopback port {port} also used by {subscriber_ports[port]}",
                field=f"{field_prefix}.port",
                line=lines.line_for("port", index, item_line),
            )

        processing_budget_ms = _require_positive_int(
            _require_key(
                item,
                "processing_budget_ms",
                f"{field_prefix}.processing_budget_ms",
                item_line,
            ),
            f"{field_prefix}.processing_budget_ms",
            lines.line_for("processing_budget_ms", index, item_line),
        )
        lease_limit = _require_positive_int(
            _require_key(
                item, "lease_limit", f"{field_prefix}.lease_limit", item_line
            ),
            f"{field_prefix}.lease_limit",
            lines.line_for("lease_limit", index, item_line),
        )
        seed = _require_int(
            _require_key(item, "seed", f"{field_prefix}.seed", item_line),
            f"{field_prefix}.seed",
            lines.line_for("seed", index, item_line),
        )

        topics: tuple[str, ...] = ()
        if "topics" in item:
            raw_topics = _require_list(
                item["topics"],
                f"{field_prefix}.topics",
                lines.line_for("topics", index, item_line),
            )
            parsed_topics: list[str] = []
            for topic_index, raw_topic in enumerate(raw_topics):
                parsed_topics.append(
                    _require_string(
                        raw_topic,
                        f"{field_prefix}.topics[{topic_index}]",
                        lines.line_for("topics", index, item_line),
                    )
                )
            topics = tuple(parsed_topics)

        subscriber_ids.add(subscriber_id)
        subscriber_ports[port] = subscriber_id
        subscribers.append(
            Subscriber(
                subscriber_id=subscriber_id,
                host=host,
                port=port,
                processing_budget_ms=processing_budget_ms,
                lease_limit=lease_limit,
                seed=seed,
                topics=topics,
                raw=item,
            )
        )

    channels: list[Channel] = []
    channel_ids: set[str] = set()
    channel_topics: dict[str, str] = {}
    for index, raw_item in enumerate(raw_channels):
        item_line = lines.line_for("channel_id", index, lines.line_for("channels"))
        item = _require_object(raw_item, f"channels[{index}]", item_line)
        field_prefix = f"channels[{index}]"

        channel_id = _require_string(
            _require_key(item, "channel_id", f"{field_prefix}.channel_id", item_line),
            f"{field_prefix}.channel_id",
            lines.line_for("channel_id", index, item_line),
        )
        if channel_id in channel_ids:
            raise SpecError(
                f"duplicate channel_id {channel_id!r}",
                field=f"{field_prefix}.channel_id",
                line=lines.line_for("channel_id", index, item_line),
            )

        topic = _require_string(
            _require_key(item, "topic", f"{field_prefix}.topic", item_line),
            f"{field_prefix}.topic",
            lines.line_for("topic", index, item_line),
        )
        if topic in channel_topics:
            raise SpecError(
                f"topic {topic!r} is already owned by channel {channel_topics[topic]}",
                field=f"{field_prefix}.topic",
                line=lines.line_for("topic", index, item_line),
            )

        batch_size = _require_positive_int(
            _require_key(item, "batch_size", f"{field_prefix}.batch_size", item_line),
            f"{field_prefix}.batch_size",
            lines.line_for("batch_size", index, item_line),
        )
        send_rate_limit = _require_positive_int(
            _require_key(
                item,
                "send_rate_limit",
                f"{field_prefix}.send_rate_limit",
                item_line,
            ),
            f"{field_prefix}.send_rate_limit",
            lines.line_for("send_rate_limit", index, item_line),
        )
        subscriber_quota = _require_positive_int(
            _require_key(
                item,
                "subscriber_quota",
                f"{field_prefix}.subscriber_quota",
                item_line,
            ),
            f"{field_prefix}.subscriber_quota",
            lines.line_for("subscriber_quota", index, item_line),
        )

        channel_ids.add(channel_id)
        channel_topics[topic] = channel_id
        channels.append(
            Channel(
                channel_id=channel_id,
                topic=topic,
                batch_size=batch_size,
                send_rate_limit=send_rate_limit,
                subscriber_quota=subscriber_quota,
                raw=item,
            )
        )

    known_topics = set(channel_topics)
    for index, subscriber in enumerate(subscribers):
        for topic in subscriber.topics:
            if topic != "*" and topic not in known_topics:
                raise SpecError(
                    f"subscriber topic {topic!r} is not declared by any channel",
                    field=f"subscribers[{index}].topics",
                    line=lines.line_for("topics", index, lines.line_for("subscriber_id", index)),
                )

    for index, channel in enumerate(channels):
        eligible = [
            subscriber
            for subscriber in subscribers
            if subscriber.accepts_topic(channel.topic)
        ]
        if not eligible:
            raise SpecError(
                f"channel topic {channel.topic!r} has no eligible subscribers",
                field=f"channels[{index}].subscriber_quota",
                line=lines.line_for("subscriber_quota", index),
            )

    declaration = Declaration(
        declaration_version=declaration_version,
        global_lease_limit=global_lease_limit,
        handshake_timeout_ms=handshake_timeout_ms,
        subscribers=tuple(subscribers),
        channels=tuple(channels),
        source_path=str(source_path),
        raw=root,
    )
    return declaration


def _event_value(
    item: dict[str, Any],
    names: Iterable[str],
    field: str,
    line: int,
) -> Any:
    for name in names:
        if name in item:
            return item[name]
    raise SpecError(f"missing required field {field}", field=field, line=line)


def validate_events_lines(
    lines: Iterable[str],
    *,
    declaration: Declaration,
    events_path: str,
) -> tuple[Event, ...]:
    topic_channels = declaration.channel_by_topic()
    events: list[Event] = []
    previous_ts: int | None = None
    expected_seq = 1
    saw_nonblank = False

    for source_line, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        saw_nonblank = True
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid NDJSON event JSON: {exc.msg}",
                field="$event_json",
                line=source_line,
                details={"column": int(exc.colno)},
            ) from exc
        if not isinstance(obj, dict):
            raise SpecError(
                "event line must be a JSON object", field="$event", line=source_line
            )

        seq_raw = _event_value(obj, ("seq", "event_id"), "seq", source_line)
        seq = _require_int(seq_raw, "seq", source_line)
        if seq != expected_seq:
            raise SpecError(
                f"event sequence must be contiguous; expected {expected_seq}, got {seq}",
                field="seq",
                line=source_line,
            )

        ts_raw = _event_value(obj, ("ts_ms", "timestamp_ms"), "ts_ms", source_line)
        ts_ms = _require_int(ts_raw, "ts_ms", source_line)
        if ts_ms < 0:
            raise SpecError(
                "ts_ms must be a non-negative integer", field="ts_ms", line=source_line
            )
        if previous_ts is not None and ts_ms < previous_ts:
            raise SpecError(
                f"ts_ms must be monotonically non-decreasing; {ts_ms} < {previous_ts}",
                field="ts_ms",
                line=source_line,
            )

        topic = _require_string(
            _event_value(obj, ("topic",), "topic", source_line), "topic", source_line
        )
        if topic not in topic_channels:
            raise SpecError(
                f"event topic {topic!r} is not declared by any channel",
                field="topic",
                line=source_line,
            )

        payload_digest = _require_string(
            _event_value(
                obj,
                ("payload_digest", "payload_summary"),
                "payload_digest",
                source_line,
            ),
            "payload_digest",
            source_line,
        )
        expected_order = _require_int(
            _event_value(obj, ("expected_order",), "expected_order", source_line),
            "expected_order",
            source_line,
        )
        if expected_order < 0:
            raise SpecError(
                "expected_order must be a non-negative integer",
                field="expected_order",
                line=source_line,
            )

        events.append(
            Event(
                seq=seq,
                ts_ms=ts_ms,
                topic=topic,
                payload_digest=payload_digest,
                expected_order=expected_order,
                line=source_line,
                raw=obj,
            )
        )
        previous_ts = ts_ms
        expected_seq += 1

    if not saw_nonblank:
        raise SpecError("event stream must not be empty", field="$event", line=1)
    return tuple(events)


def load_declaration(path: str | Path) -> Declaration:
    text = _read_text(path)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _json_error(exc) from exc
    return validate_declaration_object(obj, source_text=text, source_path=str(path))


def load_events(path: str | Path, declaration: Declaration) -> tuple[Event, ...]:
    text = _read_text(path)
    return validate_events_lines(
        text.splitlines(), declaration=declaration, events_path=str(path)
    )


def load_dataset(spec_path: str | Path, events_path: str | Path) -> Dataset:
    declaration = load_declaration(spec_path)
    spec_text = _read_text(spec_path)
    events = load_events(events_path, declaration)
    events_digest = digest_value([event.batch_content() for event in events])
    content_digest = digest_value(
        {
            "format": "eventbus.dataset.v1",
            "spec": declaration.declaration_content(),
            "events": [event.batch_content() for event in events],
        }
    )
    return Dataset(
        declaration=declaration,
        events=events,
        spec_path=str(spec_path),
        events_path=str(events_path),
        spec_text=spec_text,
        events_digest=events_digest,
        content_digest=content_digest,
    )


def dataset_digest(declaration: Declaration, events: Iterable[Event]) -> str:
    return digest_value(
        {
            "format": "eventbus.dataset.v1",
            "spec": declaration.declaration_content(),
            "events": [event.batch_content() for event in events],
        }
    )
