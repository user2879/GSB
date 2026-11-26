from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .canonical import sha256_hex
from .errors import SpecError

ACK = "ack"
DUP_ACK = "dup_ack"
LOSS = "loss"
DELAY = "delay"
REORDER = "reorder"
ENTRY_TYPES = {ACK, DUP_ACK, LOSS, DELAY, REORDER}

DEFAULT_SCALE = 1024
DEFAULT_DUPACK_THRESHOLD = 3
DEFAULT_DUPACK_CLUSTER_LIMIT = 4
TCP_MOD = 1 << 32


@dataclass(frozen=True)
class ScriptEntry:
    event_seq: int
    type: str
    line: int = 0
    ack_count: int = 1
    dup_count: int = 0
    delay_ms: int = 0
    order: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("line", None)
        data["order"] = list(self.order)
        return data


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    initial_seq: int
    event_count: int
    window_max: int
    mss: int
    scale: int
    seal_length: int
    batch_size: int = 2
    global_inflight_limit: int = 2
    connection_quota: int = 1
    initial_cwnd: int = 4
    initial_ssthresh: int = 16
    initial_rtt_ms: int = 24
    dupack_threshold: int = DEFAULT_DUPACK_THRESHOLD
    dupack_cluster_limit: int = DEFAULT_DUPACK_CLUSTER_LIMIT
    handshake_timeout_ms: int = 500
    read_timeout_ms: int = 250
    peer_host: str = "127.0.0.1"
    peer_port: int = 0
    script_version: int = 1
    scale_version: int = 1
    config_digest: str = ""
    script_digest: str = ""
    source_path: str = ""
    script_path_value: str = ""
    entries: tuple[ScriptEntry, ...] = field(default_factory=tuple)
    seed: int = 355

    @property
    def first_event_seq(self) -> int:
        return self.initial_seq

    @property
    def last_event_seq(self) -> int:
        return self.initial_seq + self.event_count - 1

    def tcp_seq(self, event_index: int) -> int:
        return (self.initial_seq + event_index * self.mss) % TCP_MOD

    def entry_for_seq(self, event_seq: int) -> ScriptEntry | None:
        for entry in self.entries:
            if entry.event_seq == event_seq:
                return entry
        return None

    def to_json(self, include_entries: bool = True) -> dict[str, Any]:
        data = {
            "session_id": self.session_id,
            "initial_seq": self.initial_seq,
            "event_count": self.event_count,
            "window_max": self.window_max,
            "mss": self.mss,
            "scale": self.scale,
            "seal_length": self.seal_length,
            "batch_size": self.batch_size,
            "global_inflight_limit": self.global_inflight_limit,
            "connection_quota": self.connection_quota,
            "initial_cwnd": self.initial_cwnd,
            "initial_ssthresh": self.initial_ssthresh,
            "initial_rtt_ms": self.initial_rtt_ms,
            "dupack_threshold": self.dupack_threshold,
            "dupack_cluster_limit": self.dupack_cluster_limit,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "read_timeout_ms": self.read_timeout_ms,
            "peer_host": self.peer_host,
            "peer_port": self.peer_port,
            "script_version": self.script_version,
            "scale_version": self.scale_version,
            "config_digest": self.config_digest,
            "script_digest": self.script_digest,
        }
        if include_entries:
            data["entries"] = [entry.to_json() for entry in self.entries]
        return data


def _line_of_pattern(text: str, pattern: str, occurrence: int = 0) -> int | None:
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(pattern, start + 1)
        if start < 0:
            return None
    return text.count("\n", 0, start) + 1


def _line_for_root_field(text: str, name: str) -> int | None:
    return _line_of_pattern(text, f'"{name}"')


def _line_for_entry_field(script_text: str, index: int, field_name: str) -> int | None:
    marker = '"event_seq"'
    positions: list[int] = []
    start = -1
    while True:
        start = script_text.find(marker, start + 1)
        if start < 0:
            break
        positions.append(start)
    if index >= len(positions):
        return None
    anchor = positions[index]
    obj_start = script_text.rfind("{", 0, anchor)
    obj_end = script_text.find("}", anchor)
    if obj_start < 0 or obj_end < 0:
        return script_text.count("\n", 0, anchor) + 1
    segment = script_text[obj_start : obj_end + 1]
    local = segment.find(f'"{field_name}"')
    if local >= 0:
        absolute = obj_start + local
    else:
        absolute = anchor
    return script_text.count("\n", 0, absolute) + 1


def _expect_int(
    data: dict[str, Any],
    name: str,
    text: str,
    *,
    source: str,
    minimum: int | None = None,
    maximum: int | None = None,
    required: bool = True,
) -> int | None:
    if name not in data:
        if required:
            raise SpecError(
                f"missing required integer field {name}",
                field_name=name,
                line=_line_for_root_field(text, name),
                details={"source": source},
            )
        return None
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(
            f"field {name} must be an integer",
            field_name=name,
            line=_line_for_root_field(text, name),
            details={"source": source, "actual_type": type(value).__name__},
        )
    if minimum is not None and value < minimum:
        raise SpecError(
            f"field {name} must be >= {minimum}",
            field_name=name,
            line=_line_for_root_field(text, name),
            details={"source": source, "value": value},
        )
    if maximum is not None and value > maximum:
        raise SpecError(
            f"field {name} must be <= {maximum}",
            field_name=name,
            line=_line_for_root_field(text, name),
            details={"source": source, "value": value},
        )
    return value


def _load_json_file(path: Path) -> tuple[Any, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read JSON file {path}: {exc}", details={"path": str(path)}) from exc
    try:
        return json.loads(text), text
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field_name="$",
            line=exc.lineno,
            details={"path": str(path), "column": exc.colno},
        ) from exc


def _parse_entries(raw_script: Any, text: str, source: str, spec_values: dict[str, int]) -> tuple[ScriptEntry, ...]:
    if not isinstance(raw_script, dict):
        raise SpecError("script root must be a JSON object", field_name="$", line=1, details={"source": source})
    raw_entries = raw_script.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise SpecError(
            "script field entries must be a non-empty array",
            field_name="entries",
            line=_line_for_root_field(text, "entries"),
            details={"source": source},
        )

    initial_seq = spec_values["initial_seq"]
    event_count = spec_values["event_count"]
    window_max = spec_values["window_max"]
    last_seq = initial_seq + event_count - 1
    entries: list[ScriptEntry] = []
    seen: set[int] = set()
    previous: int | None = None

    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise SpecError(
                "each script entry must be an object",
                field_name=f"entries[{index}]",
                line=_line_for_entry_field(text, index, "event_seq") or index + 1,
                details={"source": source},
            )

        def fail(message: str, field_name: str, *, fallback: str = "event_seq", **details: Any) -> None:
            raise SpecError(
                message,
                field_name=f"entries[{index}].{field_name}",
                line=_line_for_entry_field(text, index, field_name)
                or _line_for_entry_field(text, index, fallback),
                event_seq=raw.get("event_seq") if isinstance(raw.get("event_seq"), int) else None,
                details={"source": source, "entry_index": index, **details},
            )

        event_seq = raw.get("event_seq")
        if isinstance(event_seq, bool) or not isinstance(event_seq, int):
            fail("event_seq must be an integer", "event_seq", actual_type=type(event_seq).__name__)
        assert isinstance(event_seq, int)
        if event_seq < initial_seq or event_seq > last_seq:
            fail(
                f"event_seq must be within [{initial_seq}, {last_seq}]",
                "event_seq",
                value=event_seq,
            )
        if event_seq in seen:
            fail("duplicate event_seq is not allowed", "event_seq", value=event_seq)
        if previous is not None and event_seq <= previous:
            fail("event_seq values must be strictly increasing", "event_seq", value=event_seq)
        seen.add(event_seq)
        previous = event_seq

        entry_type = raw.get("type")
        if not isinstance(entry_type, str) or entry_type not in ENTRY_TYPES:
            fail(
                f"type must be one of {sorted(ENTRY_TYPES)}",
                "type",
                actual_type=type(entry_type).__name__,
            )
        assert isinstance(entry_type, str)

        ack_count = raw.get("ack_count", 1)
        if isinstance(ack_count, bool) or not isinstance(ack_count, int) or ack_count < 0:
            fail("ack_count must be a non-negative integer", "ack_count", value=ack_count)
        if entry_type == ACK and ack_count < 1:
            fail("ack entry requires ack_count >= 1", "ack_count", value=ack_count)

        dup_count = raw.get("dup_count", 0)
        if isinstance(dup_count, bool) or not isinstance(dup_count, int) or dup_count < 0:
            fail("dup_count must be a non-negative integer", "dup_count", value=dup_count)
        if dup_count > window_max:
            fail(
                "dup_count is incompatible with window_max",
                "dup_count",
                value=dup_count,
                window_max=window_max,
            )
        if entry_type == DUP_ACK and dup_count < 1:
            fail("dup_ack entry requires dup_count >= 1", "dup_count", value=dup_count)

        delay_ms = raw.get("delay_ms", 0)
        if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 0:
            fail("delay_ms must be a non-negative integer", "delay_ms", value=delay_ms)

        order_raw = raw.get("order", ())
        order: tuple[int, ...] = ()
        if order_raw != ():
            if not isinstance(order_raw, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in order_raw
            ):
                fail("order must be an array of integers", "order")
            order = tuple(order_raw)
            if len(order) != len(set(order)):
                fail("order must not contain duplicate event sequences", "order")
            if any(item < initial_seq or item > last_seq for item in order):
                fail(f"order entries must be within [{initial_seq}, {last_seq}]", "order")

        entries.append(
            ScriptEntry(
                event_seq=event_seq,
                type=entry_type,
                line=_line_for_entry_field(text, index, "event_seq") or 0,
                ack_count=ack_count,
                dup_count=dup_count,
                delay_ms=delay_ms,
                order=order,
            )
        )
    return tuple(entries)


def load_session(path: str | Path) -> SessionSpec:
    path = Path(path).resolve()
    data, text = _load_json_file(path)
    if not isinstance(data, dict):
        raise SpecError("session config root must be a JSON object", field_name="$", line=1)

    source = str(path)
    session_id = data.get("session_id", path.stem)
    if not isinstance(session_id, str) or not session_id:
        raise SpecError(
            "session_id must be a non-empty string",
            field_name="session_id",
            line=_line_for_root_field(text, "session_id"),
        )

    initial_seq = _expect_int(data, "initial_seq", text, source=source, minimum=0, maximum=TCP_MOD - 1) or 0
    event_count = _expect_int(data, "event_count", text, source=source, minimum=1, maximum=100000) or 1
    window_max = _expect_int(data, "window_max", text, source=source, minimum=1, maximum=65535) or 1
    mss = _expect_int(data, "mss", text, source=source, minimum=1, maximum=65495) or 1460
    scale = _expect_int(data, "scale", text, source=source, minimum=1, maximum=10_000_000) or DEFAULT_SCALE
    seal_length = _expect_int(data, "seal_length", text, source=source, minimum=1, maximum=event_count) or 4
    seed_value = _expect_int(data, "seed", text, source=source, minimum=0, maximum=2**32 - 1, required=False)
    seed = 355 if seed_value is None else seed_value
    batch_size = _expect_int(data, "batch_size", text, source=source, minimum=1, maximum=seal_length, required=False) or min(2, seal_length)
    if seal_length % batch_size != 0:
        raise SpecError(
            "seal_length must be an integer multiple of batch_size",
            field_name="seal_length",
            line=_line_for_root_field(text, "seal_length"),
            details={"seal_length": seal_length, "batch_size": batch_size},
        )

    global_inflight = _expect_int(data, "global_inflight_limit", text, source=source, minimum=1, required=False) or 2
    quota = _expect_int(data, "connection_quota", text, source=source, minimum=1, required=False) or 1
    if quota > global_inflight:
        raise SpecError(
            "connection_quota must be <= global_inflight_limit",
            field_name="connection_quota",
            line=_line_for_root_field(text, "connection_quota"),
            details={"connection_quota": quota, "global_inflight_limit": global_inflight},
        )

    initial_cwnd = _expect_int(data, "initial_cwnd", text, source=source, minimum=1, maximum=window_max, required=False) or min(4, window_max)
    initial_ssthresh = _expect_int(data, "initial_ssthresh", text, source=source, minimum=1, required=False) or window_max
    initial_rtt_ms = _expect_int(data, "initial_rtt_ms", text, source=source, minimum=1, maximum=60000, required=False) or 24
    dupack_threshold = _expect_int(data, "dupack_threshold", text, source=source, minimum=1, maximum=window_max, required=False) or 3
    cluster_limit = _expect_int(data, "dupack_cluster_limit", text, source=source, minimum=2, maximum=window_max + 1, required=False) or 4
    handshake_timeout = _expect_int(data, "handshake_timeout_ms", text, source=source, minimum=1, maximum=60000, required=False) or 500
    read_timeout = _expect_int(data, "read_timeout_ms", text, source=source, minimum=1, maximum=60000, required=False) or 250
    peer_host = data.get("peer_host", "127.0.0.1")
    if peer_host != "127.0.0.1":
        raise SpecError(
            "peer_host must be 127.0.0.1 for this local deterministic task",
            field_name="peer_host",
            line=_line_for_root_field(text, "peer_host"),
            details={"value": peer_host},
        )
    peer_port = _expect_int(data, "peer_port", text, source=source, minimum=0, maximum=65535, required=False) or 0
    script_version = _expect_int(data, "script_version", text, source=source, minimum=1, required=False) or 1
    scale_version = _expect_int(data, "scale_version", text, source=source, minimum=1, required=False) or 1

    script_file = data.get("script_file")
    if script_file is not None and not isinstance(script_file, str):
        raise SpecError("script_file must be a string", field_name="script_file", line=_line_for_root_field(text, "script_file"))

    if script_file:
        script_path = (path.parent / script_file).resolve()
        script_data, script_text = _load_json_file(script_path)
        script_source = str(script_path)
        if isinstance(script_data, dict):
            script_version = script_data.get("script_version", script_version)
            if isinstance(script_version, bool) or not isinstance(script_version, int) or script_version < 1:
                raise SpecError(
                    "script_version must be a positive integer",
                    field_name="script_version",
                    line=_line_for_root_field(script_text, "script_version"),
                    details={"source": script_source},
                )
        spec_values = {
            "initial_seq": initial_seq,
            "event_count": event_count,
            "window_max": window_max,
        }
        entries = _parse_entries(script_data, script_text, script_source, spec_values)
        script_digest = sha256_hex(script_text.encode("utf-8"))
    else:
        script_data = data
        script_text = text
        entries = _parse_entries(data, text, source, {
            "initial_seq": initial_seq,
            "event_count": event_count,
            "window_max": window_max,
        })
        script_digest = sha256_hex(canonical_script_digest(entries, script_version))

    config_digest = sha256_hex(text.encode("utf-8"))
    return SessionSpec(
        session_id=session_id,
        initial_seq=initial_seq,
        event_count=event_count,
        window_max=window_max,
        mss=mss,
        scale=scale,
        seal_length=seal_length,
        batch_size=batch_size,
        global_inflight_limit=global_inflight,
        connection_quota=quota,
        initial_cwnd=initial_cwnd,
        initial_ssthresh=initial_ssthresh,
        initial_rtt_ms=initial_rtt_ms,
        dupack_threshold=dupack_threshold,
        dupack_cluster_limit=cluster_limit,
        handshake_timeout_ms=handshake_timeout,
        read_timeout_ms=read_timeout,
        peer_host=peer_host,
        peer_port=peer_port,
        script_version=script_version,
        scale_version=scale_version,
        config_digest=config_digest,
        script_digest=script_digest,
        source_path=str(path),
        script_path_value=script_file or "",
        entries=entries,
        seed=seed,
    )


def canonical_script_digest(entries: tuple[ScriptEntry, ...], version: int) -> bytes:
    import json

    return json.dumps(
        {"script_version": version, "entries": [e.to_json() for e in entries]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
