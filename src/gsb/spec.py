"""Declarative specification loading with strict first-failure validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SpecError

HOST_LITERAL = "127.0.0.1"


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class Subscriber:
    subscriber_id: str
    host: str
    port: int
    budget_ms: int
    inflight_event_limit: int
    seed: int
    topics: tuple[str, ...]


@dataclass(frozen=True)
class Channel:
    channel_id: str
    topic: str
    batch_size: int
    rate_limit: int


@dataclass(frozen=True)
class Event:
    seq: int
    timestamp_ms: int
    topic: str
    payload: int


@dataclass(frozen=True)
class Spec:
    subscriber_limit: int
    error_margin_ms: int
    subscribers: tuple[Subscriber, ...]
    channels: tuple[Channel, ...]
    events: tuple[Event, ...]
    spec_digest: str
    topics: tuple[str, ...] = field(default_factory=tuple)

    def subscriber(self, subscriber_id: str) -> Subscriber:
        for sub in self.subscribers:
            if sub.subscriber_id == subscriber_id:
                return sub
        raise KeyError(subscriber_id)


def _fail(field_name, message, entry):
    raise SpecError(message, entry=entry, field=field_name)


def _require_str(mapping, key, entry):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        _fail(key, f"field '{key}' must be a non-empty string", entry)
    return value


def _validate_subscribers(raw, used_ports):
    subscribers = []
    seen_ids = set()
    for index, item in enumerate(raw):
        entry = f"subscribers[{index}]"
        if not isinstance(item, dict):
            _fail("<item>", "subscriber entry must be an object", entry)
        sid = _require_str(item, "subscriber_id", entry)
        if sid in seen_ids:
            _fail("subscriber_id", f"duplicate subscriber_id '{sid}'", entry)
        seen_ids.add(sid)

        host = item.get("host")
        if host != HOST_LITERAL:
            _fail("host", f"host must be the literal {HOST_LITERAL}", entry)

        port = item.get("port")
        if not _is_positive_int(port):
            _fail("port", "port must be a positive integer", entry)
        if port in used_ports:
            _fail("port", f"port {port} is already bound to another subscriber", entry)
        used_ports.add(port)

        for key in ("budget_ms", "inflight_event_limit"):
            if not _is_positive_int(item.get(key)):
                _fail(key, f"{key} must be a positive integer", entry)
        if not _is_nonneg_int(item.get("seed")):
            _fail("seed", "seed must be a non-negative integer", entry)

        topics = item.get("topics")
        if (
            not isinstance(topics, list)
            or not topics
            or not all(isinstance(t, str) and t for t in topics)
        ):
            _fail("topics", "topics must be a non-empty list of strings", entry)
        if len(set(topics)) != len(topics):
            _fail("topics", "subscriber topics must be unique", entry)

        subscribers.append(
            Subscriber(
                subscriber_id=sid,
                host=host,
                port=port,
                budget_ms=item["budget_ms"],
                inflight_event_limit=item["inflight_event_limit"],
                seed=item["seed"],
                topics=tuple(topics),
            )
        )
    if not subscribers:
        raise SpecError(
            "at least one subscriber is required", entry="subscribers", field="<list>"
        )
    return tuple(subscribers)


def _validate_channels(raw):
    channels = []
    seen_ids = set()
    for index, item in enumerate(raw):
        entry = f"channels[{index}]"
        if not isinstance(item, dict):
            _fail("<item>", "channel entry must be an object", entry)
        cid = _require_str(item, "channel_id", entry)
        if cid in seen_ids:
            _fail("channel_id", f"duplicate channel_id '{cid}'", entry)
        seen_ids.add(cid)
        topic = _require_str(item, "topic", entry)
        for key in ("batch_size", "rate_limit"):
            if not _is_positive_int(item.get(key)):
                _fail(key, f"{key} must be a positive integer", entry)
        channels.append(
            Channel(
                channel_id=cid,
                topic=topic,
                batch_size=item["batch_size"],
                rate_limit=item["rate_limit"],
            )
        )
    if not channels:
        raise SpecError(
            "at least one channel is required", entry="channels", field="<list>"
        )
    return tuple(channels)


def _load_events(events_path, known_topics):
    events = []
    last_seq = None
    last_ts = None
    with open(events_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            entry = f"events.line[{line_number}]"
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _fail("<json>", f"invalid NDJSON: {exc.msg}", entry)
            if not isinstance(item, dict):
                _fail("<item>", "event must be an object", entry)
            seq = item.get("seq")
            if not _is_nonneg_int(seq):
                _fail("seq", "seq must be a non-negative integer", entry)
            if last_seq is not None and seq < last_seq:
                _fail("seq", f"event seq {seq} is less than previous {last_seq}", entry)
            last_seq = seq

            ts = item.get("timestamp_ms")
            if not _is_nonneg_int(ts):
                _fail(
                    "timestamp_ms", "timestamp_ms must be a non-negative integer", entry
                )
            if last_ts is not None and ts < last_ts:
                _fail(
                    "timestamp_ms",
                    f"timestamp_ms {ts} is less than previous {last_ts}",
                    entry,
                )
            last_ts = ts
            topic = _require_str(item, "topic", entry)
            if topic not in known_topics:
                _fail(
                    "topic",
                    f"event topic '{topic}' is not declared by any channel",
                    entry,
                )
            payload = item.get("payload")
            if not isinstance(payload, int) or isinstance(payload, bool):
                _fail("payload", "payload must be an integer", entry)
            events.append(Event(seq=seq, timestamp_ms=ts, topic=topic, payload=payload))
    if not events:
        raise SpecError(
            "event stream must not be empty", entry="events", field="<stream>"
        )
    return tuple(events)


def load_spec(spec_path, events_path=None) -> Spec:
    spec_path = Path(spec_path)
    with open(spec_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SpecError("spec root must be an object", entry="<root>", field="<root>")

    subscriber_limit = raw.get("subscriber_limit", 3)
    if not _is_positive_int(subscriber_limit):
        _fail(
            "subscriber_limit", "subscriber_limit must be a positive integer", "<root>"
        )
    error_margin = raw.get("error_margin_ms", 2)
    if not _is_nonneg_int(error_margin):
        _fail(
            "error_margin_ms",
            "error_margin_ms must be a non-negative integer",
            "<root>",
        )

    used_ports: set[int] = set()
    subscribers = _validate_subscribers(raw.get("subscribers", []), used_ports)
    channels = _validate_channels(raw.get("channels", []))

    channel_topics = {channel.topic for channel in channels}
    subscribed_topics = {topic for sub in subscribers for topic in sub.topics}
    unknown = channel_topics - subscribed_topics
    if unknown:
        raise SpecError(
            f"channel topics have no subscriber: {sorted(unknown)}",
            entry="channels",
            field="topic",
        )

    if events_path is None:
        events_path = spec_path.parent / raw.get("events_file", "events.ndjson")
    events = _load_events(Path(events_path), channel_topics)

    spec_digest = _spec_digest(
        subscribers, channels, events, subscriber_limit, error_margin
    )
    return Spec(
        subscriber_limit=subscriber_limit,
        error_margin_ms=error_margin,
        subscribers=subscribers,
        channels=channels,
        events=events,
        spec_digest=spec_digest,
        topics=tuple(sorted(channel_topics)),
    )


def _spec_digest(subscribers, channels, events, subscriber_limit, error_margin):
    from .digest import sha256_hex

    payload = {
        "subscriber_limit": subscriber_limit,
        "error_margin_ms": error_margin,
        "subscribers": [sub.__dict__ for sub in subscribers],
        "channels": [chan.__dict__ for chan in channels],
        "events": [evt.__dict__ for evt in events],
    }
    return sha256_hex(payload)
