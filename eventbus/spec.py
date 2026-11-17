"""Loading and strong validation of event-source and subscriber specs.

Any validation failure rejects the whole load with E_SPEC and reports the
offending field path (e.g. ``sources[1].event_seqs``).
"""

import json
from dataclasses import dataclass

from .errors import BusError, E_SPEC


@dataclass(frozen=True)
class Event:
    source_id: str
    seq: int
    event_type: str
    ts_ms: int


@dataclass(frozen=True)
class EventSource:
    source_id: str
    event_types: tuple
    event_count: int
    event_seqs: tuple
    first_event_ms: int
    interval_ms: int

    def events(self):
        produced = []
        for index, seq in enumerate(self.event_seqs):
            produced.append(Event(
                source_id=self.source_id,
                seq=seq,
                event_type=self.event_types[index % len(self.event_types)],
                ts_ms=self.first_event_ms + index * self.interval_ms,
            ))
        return produced


@dataclass(frozen=True)
class Subscriber:
    subscriber_id: str
    event_types: tuple
    timeout_ms: int


@dataclass(frozen=True)
class Spec:
    sources: tuple
    subscribers: tuple
    set_id: str
    set_version: int
    events: tuple
    subscriber_snapshot: str


def _fail(field, message):
    raise BusError(E_SPEC, message, field=field)


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_negative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        _fail(label, "file not found: %s" % path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(label, "file is not valid JSON: %s" % exc)
    except OSError as exc:
        _fail(label, "cannot read file: %s" % exc)


def _require_str_list(value, field, *, non_empty=True):
    if not isinstance(value, list) or (non_empty and not value):
        _fail(field, "must be a non-empty list of strings")
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            _fail("%s[%d]" % (field, index), "must be a non-empty string")
        if item in seen:
            _fail("%s[%d]" % (field, index), "duplicate entry %r" % item)
        seen.add(item)
    return tuple(value)


def _load_sources(path):
    doc = _read_json(path, "sources")
    if not isinstance(doc, dict) or not isinstance(doc.get("sources"), list) or not doc["sources"]:
        _fail("sources", "document must contain a non-empty 'sources' list")
    sources = []
    seen_ids = set()
    for index, raw in enumerate(doc["sources"]):
        base = "sources[%d]" % index
        if not isinstance(raw, dict):
            _fail(base, "source entry must be an object")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            _fail(base + ".source_id", "must be a non-empty string")
        if source_id in seen_ids:
            _fail(base + ".source_id", "duplicate source id %r" % source_id)
        seen_ids.add(source_id)
        event_types = _require_str_list(raw.get("event_types"), base + ".event_types")
        event_count = raw.get("event_count")
        if not _is_positive_int(event_count):
            _fail(base + ".event_count", "must be a positive integer")
        event_seqs = raw.get("event_seqs")
        if not isinstance(event_seqs, list) or not event_seqs:
            _fail(base + ".event_seqs", "must be a non-empty list of integers")
        for seq_index, seq in enumerate(event_seqs):
            if not _is_positive_int(seq):
                _fail("%s.event_seqs[%d]" % (base, seq_index), "must be a positive integer")
            if seq_index and seq <= event_seqs[seq_index - 1]:
                _fail("%s.event_seqs[%d]" % (base, seq_index),
                      "event seqs must be strictly increasing")
        if len(event_seqs) != event_count:
            _fail(base + ".event_seqs",
                  "seq count %d does not match event_count %d" % (len(event_seqs), event_count))
        first_event_ms = raw.get("first_event_ms")
        if not _is_non_negative_int(first_event_ms):
            _fail(base + ".first_event_ms", "must be a non-negative integer")
        interval_ms = raw.get("interval_ms")
        if not _is_positive_int(interval_ms):
            _fail(base + ".interval_ms", "must be a positive integer")
        sources.append(EventSource(
            source_id=source_id,
            event_types=event_types,
            event_count=event_count,
            event_seqs=tuple(event_seqs),
            first_event_ms=first_event_ms,
            interval_ms=interval_ms,
        ))
    return tuple(sources)


def _load_subscribers(path, supported_types):
    doc = _read_json(path, "subscribers")
    if not isinstance(doc, dict):
        _fail("subscribers", "document must be an object")
    set_id = doc.get("set_id")
    if not isinstance(set_id, str) or not set_id:
        _fail("set_id", "must be a non-empty string")
    set_version = doc.get("set_version")
    if not _is_positive_int(set_version):
        _fail("set_version", "must be a positive integer")
    raw_subs = doc.get("subscribers")
    if not isinstance(raw_subs, list) or not raw_subs:
        _fail("subscribers", "must be a non-empty list")
    subscribers = []
    seen_ids = set()
    for index, raw in enumerate(raw_subs):
        base = "subscribers[%d]" % index
        if not isinstance(raw, dict):
            _fail(base, "subscriber entry must be an object")
        subscriber_id = raw.get("subscriber_id")
        if not isinstance(subscriber_id, str) or not subscriber_id:
            _fail(base + ".subscriber_id", "must be a non-empty string")
        if subscriber_id in seen_ids:
            _fail(base + ".subscriber_id", "duplicate subscriber id %r" % subscriber_id)
        seen_ids.add(subscriber_id)
        event_types = _require_str_list(raw.get("event_types"), base + ".event_types")
        for type_index, event_type in enumerate(event_types):
            if event_type not in supported_types:
                _fail("%s.event_types[%d]" % (base, type_index),
                      "event type %r is not declared by any source" % event_type)
        timeout_ms = raw.get("timeout_ms")
        if not _is_positive_int(timeout_ms):
            _fail(base + ".timeout_ms", "must be a positive integer")
        subscribers.append(Subscriber(
            subscriber_id=subscriber_id,
            event_types=event_types,
            timeout_ms=timeout_ms,
        ))
    snapshot = json.dumps(
        {"set_id": set_id, "set_version": set_version, "subscribers": raw_subs},
        ensure_ascii=False, sort_keys=True)
    return tuple(subscribers), set_id, set_version, snapshot


def load_specs(sources_path, subscribers_path):
    """Load and validate both spec files; reject the whole load on any error."""
    sources = _load_sources(sources_path)
    supported_types = set()
    for source in sources:
        supported_types.update(source.event_types)
    subscribers, set_id, set_version, snapshot = _load_subscribers(
        subscribers_path, supported_types)
    events = []
    for source in sources:
        events.extend(source.events())
    events.sort(key=lambda event: (event.ts_ms, event.source_id, event.seq))
    return Spec(
        sources=sources,
        subscribers=subscribers,
        set_id=set_id,
        set_version=set_version,
        events=tuple(events),
        subscriber_snapshot=snapshot,
    )
