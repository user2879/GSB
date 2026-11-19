"""Session declaration and deterministic frame-script loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DeclarationError, FrameError
from .frames import (
    FLAG_BITS,
    FRAME_TYPE_CODES,
    Frame,
    HeaderField,
    PrioritySpec,
    Setting,
    decode_frame,
    encode_frame,
)
from .hpack import find_name_index
from .util import canonical_json, digest_value, is_int, read_json, require_int


SUPPORTED_FRAME_SIZES = (16384, 32768, 65535, 1048576, 16777215)
POLICIES = ("weighted_round_robin", "window_priority")
MAX_WINDOW = (1 << 31) - 1
MAX_STREAM_ID = (1 << 31) - 1


@dataclass(frozen=True)
class FlowLimits:
    connection_window: int
    stream_window: int

    def to_dict(self) -> dict[str, int]:
        return {
            "connection_window": self.connection_window,
            "stream_window": self.stream_window,
        }


@dataclass(frozen=True)
class RequestSpec:
    stream_id: int
    weight: int
    parent_id: int
    body_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "stream_id": self.stream_id,
            "weight": self.weight,
            "parent_id": self.parent_id,
            "body_bytes": self.body_bytes,
        }


@dataclass(frozen=True)
class SessionDeclaration:
    schema_version: int
    declaration_version: int
    client: FlowLimits
    server: FlowLimits
    max_concurrent_streams: int
    max_frame_size: int
    requests: tuple[RequestSpec, ...]
    source_digest: str

    @property
    def requests_by_id(self) -> dict[int, RequestSpec]:
        return {item.stream_id: item for item in self.requests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "declaration_version": self.declaration_version,
            "client": self.client.to_dict(),
            "server": self.server.to_dict(),
            "max_concurrent_streams": self.max_concurrent_streams,
            "max_frame_size": self.max_frame_size,
            "requests": [item.to_dict() for item in self.requests],
        }

    @property
    def content_digest(self) -> str:
        return digest_value(self.to_dict())


@dataclass(frozen=True)
class ScriptEvent:
    index: int
    kind: str
    label: str
    frame: Frame | None = None
    max_frames: int | None = None
    strategy: str | None = None

    def digest_payload(self) -> Any:
        if self.kind == "recv":
            return {"kind": "recv", "frame": frame_semantic_dict(self.frame)}
        if self.kind == "schedule":
            return {"kind": "schedule", "max_frames": self.max_frames}
        return {"kind": "switch_strategy", "strategy": self.strategy}


@dataclass(frozen=True)
class FrameScript:
    script_version: int
    declaration_version: int
    events: tuple[ScriptEvent, ...]
    expect_complete: bool
    source_digest: str
    event_digests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_version": self.script_version,
            "declaration_version": self.declaration_version,
            "expect_complete": self.expect_complete,
            "events": [event.digest_payload() for event in self.events],
        }

    def prefix_digest(self, count: int) -> str:
        return digest_value(list(self.event_digests[:count]))

    def slice(self, stop: int) -> "FrameScript":
        events = self.events[:stop]
        return FrameScript(
            script_version=self.script_version,
            declaration_version=self.declaration_version,
            events=events,
            expect_complete=self.expect_complete,
            source_digest=digest_value([event.digest_payload() for event in events]),
            event_digests=self.event_digests[:stop],
        )


def frame_semantic_dict(frame: Frame | None) -> dict[str, Any]:
    if frame is None:
        return {}
    result = frame.to_dict()
    result["flags_int"] = frame.flags
    if frame.raw_bytes is not None:
        result["raw_hex"] = frame.raw_bytes.hex()
    if frame.type == "DATA":
        result["body_hex"] = frame.body.hex()
    return result


def _known_keys(value: dict[str, Any], allowed: Iterable[str], path: str) -> None:
    for key in sorted(value):
        if key not in allowed:
            raise DeclarationError("unknown field", f"{path}.{key}")


def _window_block(value: Any, path: str) -> FlowLimits:
    if not isinstance(value, dict):
        raise DeclarationError("expected an object", path)
    aliases = {
        "connection_window": "connection_window",
        "initial_connection_window": "connection_window",
        "stream_window": "stream_window",
        "initial_stream_window": "stream_window",
    }
    _known_keys(value, set(aliases), path)
    normalized: dict[str, int] = {}
    for key in sorted(value):
        target = aliases[key]
        if target in normalized:
            raise DeclarationError(f"duplicate alias for {target}", f"{path}.{key}")
        normalized[target] = value[key]
    connection = require_int(normalized.get("connection_window"), f"{path}.connection_window", minimum=0, maximum=MAX_WINDOW)
    stream = require_int(normalized.get("stream_window"), f"{path}.stream_window", minimum=0, maximum=MAX_WINDOW)
    return FlowLimits(connection, stream)


def _request(value: Any, path: str) -> RequestSpec:
    if not isinstance(value, dict):
        raise DeclarationError("expected an object", path)
    aliases = {
        "stream_id": "stream_id",
        "weight": "weight",
        "parent_id": "parent_id",
        "depends_on": "parent_id",
        "body_bytes": "body_bytes",
    }
    _known_keys(value, set(aliases), path)
    normalized: dict[str, Any] = {}
    for key in sorted(value):
        target = aliases[key]
        if target in normalized:
            raise DeclarationError(f"duplicate alias for {target}", f"{path}.{key}")
        normalized[target] = value[key]
    stream_id = require_int(normalized.get("stream_id"), f"{path}.stream_id", minimum=1, maximum=MAX_STREAM_ID)
    if stream_id % 2 == 0:
        raise DeclarationError("stream_id must be an odd client-initiated stream id", f"{path}.stream_id")
    weight = require_int(normalized.get("weight"), f"{path}.weight", minimum=1, maximum=256)
    parent_id = require_int(normalized.get("parent_id"), f"{path}.parent_id", minimum=0, maximum=MAX_STREAM_ID)
    body_bytes = require_int(normalized.get("body_bytes"), f"{path}.body_bytes", minimum=0)
    return RequestSpec(stream_id, weight, parent_id, body_bytes)


def _validate_dependency_graph(requests: tuple[RequestSpec, ...]) -> None:
    by_id = {item.stream_id: item for item in requests}
    for index, item in enumerate(requests):
        if item.parent_id and item.parent_id not in by_id:
            raise DeclarationError(
                "dependency parent stream does not exist in requests",
                f"$.requests[{index}].parent_id",
            )
        if item.parent_id == item.stream_id:
            raise DeclarationError("stream cannot depend on itself", f"$.requests[{index}].parent_id")

    index_by_id = {item.stream_id: index for index, item in enumerate(requests)}
    visiting: set[int] = set()
    stack: list[int] = []
    visited: set[int] = set()

    def visit(stream_id: int) -> None:
        if stream_id in visited:
            return
        if stream_id in visiting:
            cycle_start = stack.index(stream_id) if stream_id in stack else 0
            cycle = stack[cycle_start:] + [stream_id]
            index = index_by_id[stream_id]
            raise DeclarationError(
                "priority dependency graph contains a cycle",
                f"$.requests[{index}].parent_id",
                details={"cycle": cycle},
            )
        visiting.add(stream_id)
        stack.append(stream_id)
        parent = by_id[stream_id].parent_id
        if parent:
            visit(parent)
        stack.pop()
        visiting.remove(stream_id)
        visited.add(stream_id)

    for stream_id in sorted(by_id):
        visit(stream_id)


def parse_declaration(value: Any) -> SessionDeclaration:
    if not isinstance(value, dict):
        raise DeclarationError("session declaration must be a JSON object", "$")
    allowed = {
        "schema_version",
        "declaration_version",
        "client",
        "server",
        "max_concurrent_streams",
        "max_frame_size",
        "requests",
    }
    _known_keys(value, allowed, "$")
    schema_version = require_int(value.get("schema_version"), "$.schema_version", minimum=1)
    if schema_version != 1:
        raise DeclarationError("only schema_version 1 is supported", "$.schema_version")
    declaration_version = require_int(value.get("declaration_version"), "$.declaration_version", minimum=1)
    client = _window_block(value.get("client"), "$.client")
    server = _window_block(value.get("server"), "$.server")
    max_concurrent = require_int(
        value.get("max_concurrent_streams"),
        "$.max_concurrent_streams",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    max_frame_size = require_int(value.get("max_frame_size"), "$.max_frame_size", minimum=16384, maximum=16777215)
    if max_frame_size not in SUPPORTED_FRAME_SIZES:
        supported = ", ".join(str(item) for item in SUPPORTED_FRAME_SIZES)
        raise DeclarationError(f"unsupported frame size; choose one of {supported}", "$.max_frame_size")
    requests_value = value.get("requests")
    if not isinstance(requests_value, list) or not requests_value:
        raise DeclarationError("requests must be a non-empty array", "$.requests")
    requests = tuple(_request(item, f"$.requests[{index}]") for index, item in enumerate(requests_value))
    stream_ids = [item.stream_id for item in requests]
    if len(set(stream_ids)) != len(stream_ids):
        seen: set[int] = set()
        for index, stream_id in enumerate(stream_ids):
            if stream_id in seen:
                raise DeclarationError("request stream_id must be unique", f"$.requests[{index}].stream_id")
            seen.add(stream_id)
    _validate_dependency_graph(requests)
    normalized = {
        "schema_version": schema_version,
        "declaration_version": declaration_version,
        "client": client.to_dict(),
        "server": server.to_dict(),
        "max_concurrent_streams": max_concurrent,
        "max_frame_size": max_frame_size,
        "requests": [item.to_dict() for item in sorted(requests, key=lambda item: item.stream_id)],
    }
    return SessionDeclaration(
        schema_version=schema_version,
        declaration_version=declaration_version,
        client=client,
        server=server,
        max_concurrent_streams=max_concurrent,
        max_frame_size=max_frame_size,
        requests=tuple(sorted(requests, key=lambda item: item.stream_id)),
        source_digest=digest_value(normalized),
    )


def load_declaration(path: str | Path) -> SessionDeclaration:
    return parse_declaration(read_json(path))


def declaration_from_dict(value: dict[str, Any]) -> SessionDeclaration:
    return parse_declaration(value)


def _parse_flags(value: Any, frame_type: str, path: str) -> int:
    if value is None:
        return 0
    if is_int(value):
        if not 0 <= value <= 255:
            raise DeclarationError("flags integer must be 0..255", path)
        return value
    if not isinstance(value, list):
        raise DeclarationError("flags must be an array of names or an integer", path)
    bits = FLAG_BITS.get(frame_type, {})
    result = 0
    for index, name in enumerate(value):
        if not isinstance(name, str) or name.upper() not in bits:
            raise DeclarationError(f"unsupported flag for {frame_type}", f"{path}[{index}]")
        result |= bits[name.upper()]
    return result


def _bytes_field(value: Any, path: str, *, default: bytes = b"") -> bytes:
    if value is None:
        return default
    if isinstance(value, str):
        return value.encode("utf-8")
    raise DeclarationError("expected a UTF-8 string", path)


def _header_field(value: Any, path: str) -> HeaderField:
    if is_int(value):
        if not 1 <= value <= 61:
            raise FrameError(
                f"static table index {value} is outside 1..61",
                frame_type="HEADERS",
                offset=0,
                path=path,
            )
        return HeaderField.indexed(value)
    if not isinstance(value, dict):
        raise DeclarationError("header must be a static index or an object", path)
    _known_keys(value, {"index", "name_index", "name", "value", "value_hex"}, path)
    if "index" in value:
        index = require_int(value["index"], f"{path}.index", minimum=1, maximum=61)
        return HeaderField.indexed(index)
    if "value_hex" in value:
        try:
            header_value = bytes.fromhex(value["value_hex"])
        except (ValueError, TypeError) as exc:
            raise DeclarationError("value_hex must be valid hexadecimal", f"{path}.value_hex") from exc
    else:
        if not isinstance(value.get("value"), str):
            raise DeclarationError("literal header requires a string value", f"{path}.value")
        header_value = value["value"].encode("utf-8")
    if "name_index" in value:
        name_index = require_int(value["name_index"], f"{path}.name_index", minimum=1, maximum=61)
        return HeaderField(name_index=name_index, value=header_value)
    if isinstance(value.get("name"), str):
        name = value["name"].encode("utf-8")
        return HeaderField(name=name, name_index=0, value=header_value)
    raise DeclarationError("literal header requires name_index or name", path)


def _data_payload(value: Any, path: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if not isinstance(value, dict):
        raise DeclarationError("data must be a string or object", path)
    _known_keys(value, {"utf8", "hex", "repeat"}, path)
    present = [key for key in ("utf8", "hex", "repeat") if key in value]
    if len(present) != 1:
        raise DeclarationError("data requires exactly one of utf8, hex, repeat", path)
    if "utf8" in value:
        if not isinstance(value["utf8"], str):
            raise DeclarationError("utf8 data must be a string", f"{path}.utf8")
        return value["utf8"].encode("utf-8")
    if "hex" in value:
        try:
            return bytes.fromhex(value["hex"])
        except (TypeError, ValueError) as exc:
            raise DeclarationError("hex data is invalid", f"{path}.hex") from exc
    repeat = value["repeat"]
    if not isinstance(repeat, dict):
        raise DeclarationError("repeat must be an object", f"{path}.repeat")
    _known_keys(repeat, {"byte", "length"}, f"{path}.repeat")
    byte_value = require_int(repeat.get("byte"), f"{path}.repeat.byte", minimum=0, maximum=255)
    length = require_int(repeat.get("length"), f"{path}.repeat.length", minimum=0)
    return bytes([byte_value]) * length


def frame_from_spec(
    value: Any,
    *,
    path: str,
    max_frame_size: int,
) -> Frame:
    if not isinstance(value, dict):
        raise DeclarationError("frame must be an object", path)
    if "raw_hex" in value:
        raw = value.get("raw_hex")
        if not isinstance(raw, str):
            raise DeclarationError("raw_hex must be a string", f"{path}.raw_hex")
        try:
            raw_bytes = bytes.fromhex(raw)
        except ValueError as exc:
            raise DeclarationError("raw_hex is invalid hexadecimal", f"{path}.raw_hex") from exc
        return decode_frame(raw_bytes, max_frame_size=max_frame_size, base_offset=0)
    frame_type_value = value.get("type")
    if not isinstance(frame_type_value, str):
        raise DeclarationError("frame type is required", f"{path}.type")
    frame_type = frame_type_value.upper()
    if frame_type not in FRAME_TYPE_CODES:
        raise FrameError(
            f"unsupported frame type {frame_type_value!r}",
            frame_type=frame_type,
            offset=0,
            path=f"{path}.type",
        )
    allowed = {
        "type",
        "stream_id",
        "flags",
        "settings",
        "headers",
        "priority",
        "data",
        "data_hex",
        "data_utf8",
        "increment",
        "error_code",
        "last_stream_id",
        "debug",
        "debug_hex",
    }
    _known_keys(value, allowed, path)
    stream_id = require_int(value.get("stream_id"), f"{path}.stream_id", minimum=0, maximum=MAX_STREAM_ID)
    flags = _parse_flags(value.get("flags"), frame_type, f"{path}.flags")
    frame: Frame
    if frame_type == "DATA":
        if "data" in value:
            body = _data_payload(value["data"], f"{path}.data")
        elif "data_utf8" in value:
            body = _bytes_field(value["data_utf8"], f"{path}.data_utf8")
        elif "data_hex" in value:
            if not isinstance(value["data_hex"], str):
                raise DeclarationError("data_hex must be a string", f"{path}.data_hex")
            try:
                body = bytes.fromhex(value["data_hex"])
            except ValueError as exc:
                raise DeclarationError("data_hex is invalid hexadecimal", f"{path}.data_hex") from exc
        else:
            raise DeclarationError("DATA requires data, data_utf8, or data_hex", f"{path}.data")
        frame = Frame(type="DATA", stream_id=stream_id, flags=flags, body=body)
    elif frame_type == "HEADERS":
        headers_spec = value.get("headers", [])
        if not isinstance(headers_spec, list):
            raise DeclarationError("headers must be an array", f"{path}.headers")
        headers = tuple(
            _header_field(item, f"{path}.headers[{index}]")
            for index, item in enumerate(headers_spec)
        )
        priority: PrioritySpec | None = None
        if "priority" in value:
            priority_value = value["priority"]
            if not isinstance(priority_value, dict):
                raise DeclarationError("priority must be an object", f"{path}.priority")
            _known_keys(priority_value, {"depends_on", "weight", "exclusive"}, f"{path}.priority")
            depends_on = require_int(
                priority_value.get("depends_on", 0),
                f"{path}.priority.depends_on",
                minimum=0,
                maximum=MAX_STREAM_ID,
            )
            weight = require_int(
                priority_value.get("weight", 16),
                f"{path}.priority.weight",
                minimum=1,
                maximum=256,
            )
            exclusive = priority_value.get("exclusive", False)
            if not isinstance(exclusive, bool):
                raise DeclarationError("exclusive must be a boolean", f"{path}.priority.exclusive")
            priority = PrioritySpec(depends_on=depends_on, weight=weight, exclusive=exclusive)
            flags |= FLAG_BITS["HEADERS"]["PRIORITY"]
        frame = Frame(type="HEADERS", stream_id=stream_id, flags=flags, headers=headers, priority=priority)
    elif frame_type == "SETTINGS":
        settings_spec = value.get("settings", [])
        if not isinstance(settings_spec, list):
            raise DeclarationError("settings must be an array", f"{path}.settings")
        settings: list[Setting] = []
        for index, item in enumerate(settings_spec):
            item_path = f"{path}.settings[{index}]"
            if not isinstance(item, dict):
                raise DeclarationError("setting must be an object", item_path)
            _known_keys(item, {"id", "value"}, item_path)
            identifier = require_int(item.get("id"), f"{item_path}.id", minimum=0, maximum=65535)
            setting_value = require_int(item.get("value"), f"{item_path}.value", minimum=0, maximum=(1 << 32) - 1)
            settings.append(Setting(identifier, setting_value))
        frame = Frame(type="SETTINGS", stream_id=stream_id, flags=flags, settings=tuple(settings))
    elif frame_type == "WINDOW_UPDATE":
        increment = require_int(value.get("increment"), f"{path}.increment", minimum=1, maximum=MAX_WINDOW)
        frame = Frame(type="WINDOW_UPDATE", stream_id=stream_id, flags=flags, increment=increment)
    elif frame_type == "RST_STREAM":
        error_code = require_int(value.get("error_code", 0), f"{path}.error_code", minimum=0, maximum=(1 << 32) - 1)
        frame = Frame(type="RST_STREAM", stream_id=stream_id, flags=flags, error_code=error_code)
    elif frame_type == "GOAWAY":
        last_stream_id = require_int(
            value.get("last_stream_id", 0),
            f"{path}.last_stream_id",
            minimum=0,
            maximum=MAX_STREAM_ID,
        )
        error_code = require_int(value.get("error_code", 0), f"{path}.error_code", minimum=0, maximum=(1 << 32) - 1)
        if "debug_hex" in value:
            if not isinstance(value["debug_hex"], str):
                raise DeclarationError("debug_hex must be a string", f"{path}.debug_hex")
            try:
                debug = bytes.fromhex(value["debug_hex"])
            except ValueError as exc:
                raise DeclarationError("debug_hex is invalid hexadecimal", f"{path}.debug_hex") from exc
        else:
            debug = _bytes_field(value.get("debug"), f"{path}.debug")
        frame = Frame(
            type="GOAWAY",
            stream_id=stream_id,
            flags=flags,
            last_stream_id=last_stream_id,
            error_code=error_code,
            debug_data=debug,
        )
    else:  # pragma: no cover - guarded by FRAME_TYPE_CODES
        raise AssertionError(frame_type)
    raw = encode_frame(frame, max_frame_size=max_frame_size)
    return decode_frame(raw, max_frame_size=max_frame_size)


def _script_event(value: Any, index: int, max_frame_size: int) -> ScriptEvent:
    path = f"$.events[{index}]"
    if isinstance(value, dict) and ("frame" in value or "event" in value or "kind" in value):
        event_kind = value.get("event", value.get("kind"))
        if event_kind is None and "frame" in value:
            event_kind = "recv"
        if not isinstance(event_kind, str):
            raise DeclarationError("event kind is required", f"{path}.event")
        event_kind = event_kind.lower()
        label = value.get("label", "")
        if not isinstance(label, str):
            raise DeclarationError("label must be a string", f"{path}.label")
        if event_kind in {"recv", "receive"}:
            _known_keys(value, {"event", "kind", "frame", "label"}, path)
            frame = frame_from_spec(value.get("frame"), path=f"{path}.frame", max_frame_size=max_frame_size)
            return ScriptEvent(index=index, kind="recv", label=label, frame=frame)
        if event_kind in {"schedule", "send"}:
            _known_keys(value, {"event", "kind", "max_frames", "label"}, path)
            max_frames = require_int(value.get("max_frames", 1), f"{path}.max_frames", minimum=1, maximum=100000)
            return ScriptEvent(index=index, kind="schedule", label=label, max_frames=max_frames)
        if event_kind in {"switch_strategy", "strategy"}:
            _known_keys(value, {"event", "kind", "strategy", "label"}, path)
            strategy = value.get("strategy")
            if strategy not in POLICIES:
                raise DeclarationError(f"strategy must be one of {', '.join(POLICIES)}", f"{path}.strategy")
            return ScriptEvent(index=index, kind="switch_strategy", label=label, strategy=strategy)
        raise DeclarationError("unknown event kind", f"{path}.event")
    if isinstance(value, dict) and "type" in value and "stream_id" in value:
        return ScriptEvent(
            index=index,
            kind="recv",
            label="",
            frame=frame_from_spec(value, path=path, max_frame_size=max_frame_size),
        )
    raise DeclarationError("event must declare a frame or a supported event kind", path)


def parse_script(
    value: Any,
    *,
    declaration: SessionDeclaration | None = None,
) -> FrameScript:
    max_frame_size = declaration.max_frame_size if declaration is not None else (1 << 24) - 1
    if isinstance(value, list):
        value = {
            "script_version": 1,
            "declaration_version": declaration.declaration_version if declaration else 1,
            "events": value,
        }
    if not isinstance(value, dict):
        raise DeclarationError("frame script must be a JSON object or array", "$")
    _known_keys(value, {"script_version", "declaration_version", "expect_complete", "events", "description"}, "$")
    script_version = require_int(value.get("script_version", 1), "$.script_version", minimum=1)
    if script_version != 1:
        raise DeclarationError("only script_version 1 is supported", "$.script_version")
    script_declaration_version = require_int(
        value.get("declaration_version"),
        "$.declaration_version",
        minimum=1,
    )
    if declaration is not None and script_declaration_version != declaration.declaration_version:
        raise DeclarationError(
            "script declaration_version does not match the session declaration",
            "$.declaration_version",
            details={"expected": declaration.declaration_version, "actual": script_declaration_version},
        )
    expect_complete = value.get("expect_complete", True)
    if not isinstance(expect_complete, bool):
        raise DeclarationError("expect_complete must be a boolean", "$.expect_complete")
    events_value = value.get("events")
    if not isinstance(events_value, list) or not events_value:
        raise DeclarationError("events must be a non-empty array", "$.events")
    events = tuple(
        _script_event(item, index, max_frame_size)
        for index, item in enumerate(events_value)
    )
    normalized = {
        "script_version": script_version,
        "declaration_version": script_declaration_version,
        "expect_complete": expect_complete,
        "events": [event.digest_payload() for event in events],
    }
    return FrameScript(
        script_version=script_version,
        declaration_version=script_declaration_version,
        events=events,
        expect_complete=expect_complete,
        source_digest=digest_value(normalized),
        event_digests=tuple(digest_value(event.digest_payload()) for event in events),
    )


def load_script(path: str | Path, declaration: SessionDeclaration | None = None) -> FrameScript:
    return parse_script(read_json(path), declaration=declaration)
