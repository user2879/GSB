"""Strict local JSON loading and validation for the three-stage limiter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canon import sha256_value
from .errors import DiagnosticError, spec_error


SPEC_FORMAT = "gateway-limit-spec/v1"
EVENT_FORMAT = "gateway-limit-events/v1"
LEVEL_NAMES = ("ingress", "relay", "egress")
LOOPBACK_LITERAL = "127.0.0.1"
STABLE_TIEBREAK = "level_then_seq"
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class Bucket:
    bucket_id: str
    level: int
    name: str
    host: str
    port: int
    rate_per_sec: int
    burst_capacity: int
    queue_limit: int
    packet_len: int

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "level": self.level,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "rate_per_sec": self.rate_per_sec,
            "burst_capacity": self.burst_capacity,
            "queue_limit": self.queue_limit,
            "packet_len": self.packet_len,
        }


@dataclass(frozen=True)
class DecisionPolicy:
    window_ms: int
    event_count_threshold: int
    mean_gap_threshold_ms: int
    work_tolerance_bp: int
    dominant_threshold_bp: int
    batch_size: int
    tcp_timeout_ms: int
    startup_timeout_ms: int
    stable_tiebreak: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "window_ms": self.window_ms,
            "event_count_threshold": self.event_count_threshold,
            "mean_gap_threshold_ms": self.mean_gap_threshold_ms,
            "work_tolerance_bp": self.work_tolerance_bp,
            "dominant_threshold_bp": self.dominant_threshold_bp,
            "batch_size": self.batch_size,
            "tcp_timeout_ms": self.tcp_timeout_ms,
            "startup_timeout_ms": self.startup_timeout_ms,
            "stable_tiebreak": self.stable_tiebreak,
        }


@dataclass(frozen=True)
class Spec:
    session_id: str
    version: int
    version_effective_from_seq: int
    scale_exponent: int
    decision: DecisionPolicy
    buckets: tuple[Bucket, Bucket, Bucket]
    source_file: str = ""
    raw_digest: str = ""

    @property
    def scale_factor(self) -> int:
        return 1 << self.scale_exponent

    @property
    def bucket_count(self) -> int:
        return len(self.buckets)

    def bucket_for_level(self, level: int) -> Bucket:
        if level < 1 or level > 3:
            raise ValueError(f"invalid level {level}")
        return self.buckets[level - 1]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": SPEC_FORMAT,
            "session_id": self.session_id,
            "version": self.version,
            "version_effective_from_seq": self.version_effective_from_seq,
            "scale_exponent": self.scale_exponent,
            "decision": self.decision.canonical_dict(),
            "buckets": [bucket.canonical_dict() for bucket in self.buckets],
        }

    def digest(self) -> str:
        return sha256_value(self.canonical_dict())


@dataclass(frozen=True)
class Event:
    seq: int
    arrival_ms: int

    def canonical_dict(self) -> dict[str, int]:
        return {"seq": self.seq, "arrival_ms": self.arrival_ms}


@dataclass(frozen=True)
class EventStream:
    session_id: str
    stream_version: int
    events: tuple[Event, ...]
    source_file: str = ""

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": EVENT_FORMAT,
            "session_id": self.session_id,
            "stream_version": self.stream_version,
            "events": [event.canonical_dict() for event in self.events],
        }

    def digest(self) -> str:
        return sha256_value(self.canonical_dict())

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def mean_gap(self) -> Fraction:
        if len(self.events) <= 1:
            return Fraction(0, 1)
        span = self.events[-1].arrival_ms - self.events[0].arrival_ms
        return Fraction(span, len(self.events) - 1)

    def window_count(self, window_ms: int) -> int:
        return len({event.arrival_ms // window_ms for event in self.events})


def _fail(file: str, entry: str, field: str, message: str, **details: Any) -> None:
    raise spec_error(file, entry, field, message, **details)


def _require_mapping(value: Any, file: str, entry: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(file, entry, "value", "must be an object")
    return value


def _check_keys(
    value: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
    file: str,
    entry: str,
) -> None:
    for key in sorted(set(value) - allowed):
        _fail(file, entry, key, "unsupported field")
    for key in sorted(required - set(value)):
        _fail(file, entry, key, "required field is missing")


def _int_value(
    value: Any,
    file: str,
    entry: str,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(file, entry, field, "must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        _fail(file, entry, field, f"must be >= {minimum}", value=result)
    if maximum is not None and result > maximum:
        _fail(file, entry, field, f"must be <= {maximum}", value=result)
    return result


def _string_value(
    value: Any,
    file: str,
    entry: str,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail(file, entry, field, "must be a string")
    if not allow_empty and not value:
        _fail(file, entry, field, "must not be empty")
    return value


def _load_json(path: str | Path, expected_format: str) -> tuple[Any, str]:
    file = str(Path(path))
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        _fail(file, "$", "file", "file does not exist")
    except json.JSONDecodeError as exc:
        _fail(file, "$", "json", "invalid JSON", line=exc.lineno, column=exc.colno)
    except OSError as exc:
        _fail(file, "$", "file", f"cannot read file: {exc}")
    if not isinstance(payload, dict):
        _fail(file, "$", "value", "top-level value must be an object")
    if payload.get("format") != expected_format:
        _fail(file, "$", "format", f"must equal {expected_format!r}", value=payload.get("format"))
    return payload, file


def load_spec(path: str | Path) -> Spec:
    payload, file = _load_json(path, SPEC_FORMAT)
    entry = "$"
    allowed = {
        "format",
        "session_id",
        "version",
        "version_effective_from_seq",
        "scale_exponent",
        "decision",
        "buckets",
    }
    required = allowed.copy()
    _check_keys(payload, required, allowed, file, entry)

    session_id = _string_value(payload["session_id"], file, entry, "session_id")
    if not SESSION_RE.fullmatch(session_id):
        _fail(file, entry, "session_id", "must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    version = _int_value(payload["version"], file, entry, "version", minimum=1)
    effective = _int_value(
        payload["version_effective_from_seq"],
        file,
        entry,
        "version_effective_from_seq",
        minimum=1,
    )
    exponent = _int_value(payload["scale_exponent"], file, entry, "scale_exponent", minimum=0, maximum=6)

    decision_obj = _require_mapping(payload["decision"], file, "decision")
    decision_required = {
        "window_ms",
        "event_count_threshold",
        "mean_gap_threshold_ms",
        "work_tolerance_bp",
        "dominant_threshold_bp",
        "batch_size",
        "tcp_timeout_ms",
        "startup_timeout_ms",
        "stable_tiebreak",
    }
    _check_keys(decision_obj, decision_required, decision_required, file, "decision")
    decision = DecisionPolicy(
        window_ms=_int_value(decision_obj["window_ms"], file, "decision", "window_ms", minimum=1),
        event_count_threshold=_int_value(
            decision_obj["event_count_threshold"],
            file,
            "decision",
            "event_count_threshold",
            minimum=0,
        ),
        mean_gap_threshold_ms=_int_value(
            decision_obj["mean_gap_threshold_ms"],
            file,
            "decision",
            "mean_gap_threshold_ms",
            minimum=0,
        ),
        work_tolerance_bp=_int_value(
            decision_obj["work_tolerance_bp"],
            file,
            "decision",
            "work_tolerance_bp",
            minimum=0,
            maximum=10000,
        ),
        dominant_threshold_bp=_int_value(
            decision_obj["dominant_threshold_bp"],
            file,
            "decision",
            "dominant_threshold_bp",
            minimum=1,
            maximum=10000,
        ),
        batch_size=_int_value(decision_obj["batch_size"], file, "decision", "batch_size", minimum=1),
        tcp_timeout_ms=_int_value(
            decision_obj["tcp_timeout_ms"],
            file,
            "decision",
            "tcp_timeout_ms",
            minimum=1,
        ),
        startup_timeout_ms=_int_value(
            decision_obj["startup_timeout_ms"],
            file,
            "decision",
            "startup_timeout_ms",
            minimum=1,
        ),
        stable_tiebreak=_string_value(
            decision_obj["stable_tiebreak"],
            file,
            "decision",
            "stable_tiebreak",
        ),
    )
    if decision.stable_tiebreak != STABLE_TIEBREAK:
        _fail(
            file,
            "decision",
            "stable_tiebreak",
            f"must equal {STABLE_TIEBREAK!r}",
        )

    buckets_value = payload["buckets"]
    if not isinstance(buckets_value, list):
        _fail(file, "buckets", "value", "must be an array")
    if len(buckets_value) != 3:
        _fail(file, "buckets", "length", "must contain exactly three buckets", value=len(buckets_value))

    bucket_allowed = {
        "bucket_id",
        "level",
        "name",
        "host",
        "port",
        "rate_per_sec",
        "burst_capacity",
        "queue_limit",
        "packet_len",
    }
    buckets: list[Bucket] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, bucket_value in enumerate(buckets_value):
        bucket_entry = f"buckets[{index}]"
        bucket_obj = _require_mapping(bucket_value, file, bucket_entry)
        _check_keys(bucket_obj, bucket_allowed, bucket_allowed, file, bucket_entry)
        bucket_id = _string_value(bucket_obj["bucket_id"], file, bucket_entry, "bucket_id")
        level = _int_value(bucket_obj["level"], file, bucket_entry, "level", minimum=1, maximum=3)
        expected_level = index + 1
        if level != expected_level:
            _fail(
                file,
                bucket_entry,
                "level",
                f"must be {expected_level} to preserve ingress -> relay -> egress order",
                value=level,
            )
        name = _string_value(bucket_obj["name"], file, bucket_entry, "name")
        if name != LEVEL_NAMES[index]:
            _fail(file, bucket_entry, "name", f"must be {LEVEL_NAMES[index]!r}", value=name)
        host = _string_value(bucket_obj["host"], file, bucket_entry, "host")
        if host != LOOPBACK_LITERAL:
            _fail(file, bucket_entry, "host", "must be the literal 127.0.0.1", value=host)
        port = _int_value(bucket_obj["port"], file, bucket_entry, "port", minimum=1, maximum=65535)
        rate = _int_value(bucket_obj["rate_per_sec"], file, bucket_entry, "rate_per_sec", minimum=1)
        burst = _int_value(bucket_obj["burst_capacity"], file, bucket_entry, "burst_capacity", minimum=0)
        queue = _int_value(bucket_obj["queue_limit"], file, bucket_entry, "queue_limit", minimum=0)
        packet = _int_value(bucket_obj["packet_len"], file, bucket_entry, "packet_len", minimum=0)
        if bucket_id in seen_ids:
            _fail(file, bucket_entry, "bucket_id", "bucket_id must be unique", value=bucket_id)
        if port in seen_ports:
            _fail(file, bucket_entry, "port", "port must be globally unique", value=port)
        seen_ids.add(bucket_id)
        seen_ports.add(port)
        buckets.append(
            Bucket(
                bucket_id=bucket_id,
                level=level,
                name=name,
                host=host,
                port=port,
                rate_per_sec=rate,
                burst_capacity=burst,
                queue_limit=queue,
                packet_len=packet,
            )
        )

    spec = Spec(
        session_id=session_id,
        version=version,
        version_effective_from_seq=effective,
        scale_exponent=exponent,
        decision=decision,
        buckets=(buckets[0], buckets[1], buckets[2]),
        source_file=file,
    )
    return Spec(**{**spec.__dict__, "raw_digest": spec.digest()})


def load_events(path: str | Path) -> EventStream:
    payload, file = _load_json(path, EVENT_FORMAT)
    entry = "$"
    required = {"format", "session_id", "stream_version", "events"}
    _check_keys(payload, required, required, file, entry)
    session_id = _string_value(payload["session_id"], file, entry, "session_id")
    if not SESSION_RE.fullmatch(session_id):
        _fail(file, entry, "session_id", "must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    stream_version = _int_value(payload["stream_version"], file, entry, "stream_version", minimum=1)
    events_value = payload["events"]
    if not isinstance(events_value, list):
        _fail(file, "events", "value", "must be an array")
    if not events_value:
        _fail(file, "events", "length", "must contain at least one event")

    event_allowed = {"seq", "arrival_ms"}
    events: list[Event] = []
    previous_arrival: int | None = None
    for index, event_value in enumerate(events_value):
        event_entry = f"events[{index}]"
        event_obj = _require_mapping(event_value, file, event_entry)
        _check_keys(event_obj, event_allowed, event_allowed, file, event_entry)
        seq = _int_value(event_obj["seq"], file, event_entry, "seq", minimum=1)
        expected_seq = index + 1
        if seq != expected_seq:
            _fail(
                file,
                event_entry,
                "seq",
                f"arrival sequence must be contiguous from 1; expected {expected_seq}",
                value=seq,
            )
        arrival_ms = _int_value(event_obj["arrival_ms"], file, event_entry, "arrival_ms", minimum=0)
        if previous_arrival is not None and arrival_ms < previous_arrival:
            _fail(
                file,
                event_entry,
                "arrival_ms",
                "arrival_ms must be monotonically nondecreasing",
                previous=previous_arrival,
                value=arrival_ms,
            )
        previous_arrival = arrival_ms
        events.append(Event(seq=seq, arrival_ms=arrival_ms))
    return EventStream(
        session_id=session_id,
        stream_version=stream_version,
        events=tuple(events),
        source_file=file,
    )


def load_inputs(spec_path: str | Path, events_path: str | Path) -> tuple[Spec, EventStream]:
    spec = load_spec(spec_path)
    events = load_events(events_path)
    if spec.session_id != events.session_id:
        raise spec_error(
            str(Path(events_path)),
            "$",
            "session_id",
            "event stream session_id does not match the declaration",
            spec_session_id=spec.session_id,
            event_session_id=events.session_id,
        )
    if spec.version_effective_from_seq > events.event_count + 1:
        raise spec_error(
            str(Path(spec_path)),
            "$",
            "version_effective_from_seq",
            "must be at most event_count + 1",
            value=spec.version_effective_from_seq,
            event_count=events.event_count,
        )
    return spec, events


def describe_inputs(spec: Spec, events: EventStream) -> dict[str, Any]:
    return {
        "spec": spec.canonical_dict(),
        "spec_digest": spec.digest(),
        "events": {
            "format": EVENT_FORMAT,
            "session_id": events.session_id,
            "stream_version": events.stream_version,
            "event_count": events.event_count,
            "first_arrival_ms": events.events[0].arrival_ms,
            "last_arrival_ms": events.events[-1].arrival_ms,
            "mean_gap": str(events.mean_gap),
            "window_count": events.window_count(spec.decision.window_ms),
            "digest": events.digest(),
        },
    }


def error_as_json(error: DiagnosticError) -> str:
    return json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

