"""Input schema parsing and validation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from .errors import EngineError
from .util import is_int, is_number, read_json, relative_or_absolute


STYLE_FIELDS = {
    "font_size",
    "line_height",
    "max_lines",
    "padding",
    "contrast_enhance",
    "mask_opacity",
}


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class Anchor:
    x: float
    y: float
    horizontal: str = "center"
    vertical: str = "center"
    source: str = "point"


@dataclass(frozen=True)
class Event:
    id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: str
    language: str
    max_chars_per_sec: int
    critical_sound: bool
    anchor: Anchor

    def with_text(self, text: str) -> "Event":
        return Event(**{**self.__dict__, "text": text})

    def with_time(self, start_ms: int, end_ms: int) -> "Event":
        return Event(**{**self.__dict__, "start_ms": start_ms, "end_ms": end_ms})


@dataclass(frozen=True)
class Style:
    font_size: float
    line_height: float
    max_lines: int
    padding: float
    contrast_enhance: bool
    mask_opacity: float

    def with_field(self, field_name: str, value: Any) -> "Style":
        return Style(**{**self.__dict__, field_name: value})


@dataclass(frozen=True)
class FrameConfig:
    width: int
    height: int
    frame_ms: int
    background: int


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int = 1
    memory_budget_bytes: int = 64 * 1024 * 1024
    block_ms: int = 500
    state_dir: Path = dc_field(default_factory=lambda: Path("state"))
    output_dir: Path = dc_field(default_factory=lambda: Path("output"))


@dataclass(frozen=True)
class Edit:
    index: int
    op: str
    event_id: str | None = None
    field: str | None = None
    value: Any = None
    start_ms: int | None = None
    end_ms: int | None = None
    raw: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class Spec:
    schema_version: str
    frame: FrameConfig
    safe_area: Rect
    style: Style
    events: tuple[Event, ...]
    edits: tuple[Edit, ...]
    runtime: RuntimeConfig
    duration_ms: int
    source_path: Path
    raw: dict[str, Any]

    @property
    def frames_per_block(self) -> int:
        return self.runtime.block_ms // self.frame.frame_ms

    @property
    def total_frames(self) -> int:
        return math.ceil(self.duration_ms / self.frame.frame_ms)

    @property
    def total_blocks(self) -> int:
        return math.ceil(self.total_frames / self.frames_per_block)


def _schema_error(message: str, path: str, code: str = "E_SCHEMA") -> EngineError:
    return EngineError(code, message, details={"path": path})


def _required_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _schema_error("Expected an object", path)
    return value


def _required_string(
    value: Any, path: str, *, allow_empty: bool = False, code: str = "E_SCHEMA"
) -> str:
    if not isinstance(value, str):
        raise _schema_error("Expected a string", path, code)
    if not allow_empty and not value.strip():
        raise _schema_error("String must not be empty", path, code)
    return value


def _positive_finite(value: Any, path: str, code: str = "E_STYLE") -> float:
    if not is_number(value) or not math.isfinite(float(value)) or float(value) <= 0:
        raise _schema_error("Expected a finite positive number", path, code)
    return float(value)


def _nonnegative_finite(value: Any, path: str, code: str = "E_STYLE") -> float:
    if not is_number(value) or not math.isfinite(float(value)) or float(value) < 0:
        raise _schema_error("Expected a finite non-negative number", path, code)
    return float(value)


def _positive_int(value: Any, path: str, code: str) -> int:
    if not is_int(value) or value <= 0:
        raise _schema_error("Expected a positive integer", path, code)
    return value


def parse_style(value: Any, path: str = "style") -> Style:
    style = _required_object(value, path)
    font_size = _positive_finite(style.get("font_size"), f"{path}.font_size", "E_STYLE_FONT")
    line_height = _positive_finite(
        style.get("line_height"), f"{path}.line_height", "E_STYLE_LINE_HEIGHT"
    )
    max_lines = _positive_int(style.get("max_lines"), f"{path}.max_lines", "E_STYLE_MAX_LINES")
    padding = _nonnegative_finite(style.get("padding"), f"{path}.padding", "E_STYLE_PADDING")
    contrast = style.get("contrast_enhance")
    if not isinstance(contrast, bool):
        raise _schema_error(
            "contrast_enhance must be a boolean",
            f"{path}.contrast_enhance",
            "E_STYLE_CONTRAST",
        )
    opacity = style.get("mask_opacity")
    if (
        not is_number(opacity)
        or not math.isfinite(float(opacity))
        or not 0.0 <= float(opacity) <= 1.0
    ):
        raise _schema_error(
            "mask_opacity must be in [0, 1]", f"{path}.mask_opacity", "E_STYLE_MASK"
        )
    return Style(
        font_size=font_size,
        line_height=line_height,
        max_lines=max_lines,
        padding=padding,
        contrast_enhance=contrast,
        mask_opacity=float(opacity),
    )


_NAMED_ANCHORS = {
    "top_left": ("left", "top"),
    "top_center": ("center", "top"),
    "top_right": ("right", "top"),
    "middle_left": ("left", "center"),
    "center": ("center", "center"),
    "middle_right": ("right", "center"),
    "bottom_left": ("left", "bottom"),
    "bottom_center": ("center", "bottom"),
    "bottom_right": ("right", "bottom"),
}


def parse_anchor(value: Any, safe: Rect, path: str) -> Anchor:
    if isinstance(value, str):
        if value not in _NAMED_ANCHORS:
            raise _schema_error("Unknown named anchor", path, "E_ANCHOR")
        horizontal, vertical = _NAMED_ANCHORS[value]
        x = {
            "left": float(safe.x),
            "center": safe.x + safe.width / 2.0,
            "right": float(safe.right),
        }[horizontal]
        y = {
            "top": float(safe.y),
            "center": safe.y + safe.height / 2.0,
            "bottom": float(safe.bottom),
        }[vertical]
        anchor = Anchor(x, y, horizontal, vertical, value)
    else:
        obj = _required_object(value, path)
        x = obj.get("x")
        y = obj.get("y")
        if not is_number(x) or not is_number(y) or not math.isfinite(float(x) + float(y)):
            raise _schema_error("Anchor x/y must be finite numbers", path, "E_ANCHOR")
        horizontal = obj.get("horizontal", "center")
        vertical = obj.get("vertical", "center")
        if horizontal not in {"left", "center", "right"}:
            raise _schema_error("Invalid horizontal anchor", f"{path}.horizontal", "E_ANCHOR")
        if vertical not in {"top", "center", "bottom"}:
            raise _schema_error("Invalid vertical anchor", f"{path}.vertical", "E_ANCHOR")
        anchor = Anchor(float(x), float(y), horizontal, vertical, "point")

    if not safe.contains_point(anchor.x, anchor.y):
        raise EngineError(
            "E_ANCHOR",
            "Screen anchor must lie inside the safe area",
            details={
                "path": path,
                "anchor": [anchor.x, anchor.y],
                "safe_area": safe.as_list(),
            },
        )
    return anchor


def parse_event(value: Any, safe: Rect, index: int) -> Event:
    path = f"events[{index}]"
    obj = _required_object(value, path)
    event_id = _required_string(obj.get("id"), f"{path}.id")
    start = obj.get("start_ms")
    end = obj.get("end_ms")
    if not is_int(start) or start < 0:
        raise _schema_error("start_ms must be a non-negative integer", f"{path}.start_ms", "E_TIME")
    if not is_int(end) or end < 0:
        raise _schema_error("end_ms must be a non-negative integer", f"{path}.end_ms", "E_TIME")
    if end <= start:
        raise _schema_error("end_ms must be greater than start_ms", path, "E_TIME")
    text = _required_string(obj.get("text"), f"{path}.text", allow_empty=False, code="E_TEXT")
    speaker = _required_string(obj.get("speaker"), f"{path}.speaker")
    language = _required_string(obj.get("language"), f"{path}.language")
    rate = _positive_int(obj.get("max_chars_per_sec"), f"{path}.max_chars_per_sec", "E_RATE")
    critical = obj.get("critical_sound")
    if not isinstance(critical, bool):
        raise _schema_error(
            "critical_sound must be a boolean", f"{path}.critical_sound", "E_CRITICAL"
        )
    anchor = parse_anchor(obj.get("anchor"), safe, f"{path}.anchor")
    return Event(
        id=event_id,
        start_ms=start,
        end_ms=end,
        text=text,
        speaker=speaker,
        language=language,
        max_chars_per_sec=rate,
        critical_sound=critical,
        anchor=anchor,
    )


def _validate_event(event: Event, safe: Rect, path: str) -> None:
    if event.start_ms < 0 or event.end_ms <= event.start_ms:
        raise _schema_error("Invalid event time window", path, "E_TIME")
    if not event.text.strip():
        raise _schema_error("Event text must not be empty", path, "E_TEXT")
    if event.max_chars_per_sec <= 0:
        raise _schema_error("Reading rate must be positive", path, "E_RATE")
    if not safe.contains_point(event.anchor.x, event.anchor.y):
        raise _schema_error("Anchor lies outside the safe area", path, "E_ANCHOR")


def parse_edit(value: Any, index: int) -> Edit:
    path = f"edits[{index}]"
    obj = _required_object(value, path)
    op = _required_string(obj.get("op"), f"{path}.op")
    if op == "set_text":
        event_id = _required_string(obj.get("event_id"), f"{path}.event_id")
        text = _required_string(obj.get("text"), f"{path}.text", code="E_TEXT")
        return Edit(index, op, event_id=event_id, value=text, raw=copy.deepcopy(obj))
    if op == "set_time":
        event_id = _required_string(obj.get("event_id"), f"{path}.event_id")
        start = obj.get("start_ms")
        end = obj.get("end_ms")
        if not is_int(start) or start < 0 or not is_int(end) or end < 0 or end <= start:
            raise _schema_error("Invalid set_time values", path, "E_EDIT")
        return Edit(
            index,
            op,
            event_id=event_id,
            start_ms=start,
            end_ms=end,
            raw=copy.deepcopy(obj),
        )
    if op == "set_style":
        field_name = obj.get("field", obj.get("path"))
        field_name = _required_string(field_name, f"{path}.field")
        if field_name not in STYLE_FIELDS:
            raise _schema_error("Unknown style field", f"{path}.field", "E_EDIT")
        if "value" not in obj:
            raise _schema_error("Missing style value", f"{path}.value", "E_EDIT")
        return Edit(
            index,
            op,
            field=field_name,
            value=copy.deepcopy(obj["value"]),
            raw=copy.deepcopy(obj),
        )
    raise _schema_error("Unknown edit operation", f"{path}.op", "E_EDIT")


def parse_spec(data: dict[str, Any], source_path: Path) -> Spec:
    obj = _required_object(data, "$")
    schema_version = str(obj.get("schema_version", "subtitle-engine/v1"))
    if schema_version != "subtitle-engine/v1":
        raise _schema_error("Unsupported schema_version", "schema_version")

    frame_obj = _required_object(obj.get("frame"), "frame")
    width = _positive_int(frame_obj.get("width"), "frame.width", "E_FRAME")
    height = _positive_int(frame_obj.get("height"), "frame.height", "E_FRAME")
    frame_ms = _positive_int(frame_obj.get("frame_ms"), "frame.frame_ms", "E_FRAME")
    background = frame_obj.get("background", 18)
    if not is_int(background) or not 0 <= background <= 255:
        raise _schema_error("frame.background must be an integer in [0, 255]", "frame.background", "E_FRAME")

    safe_obj = _required_object(obj.get("safe_area"), "safe_area")
    safe_vals = []
    for name in ("x", "y", "width", "height"):
        val = safe_obj.get(name)
        if not is_int(val) or val < 0:
            raise _schema_error(
                f"safe_area.{name} must be a non-negative integer", f"safe_area.{name}", "E_SAFE_AREA"
            )
        safe_vals.append(val)
    safe = Rect(*safe_vals)
    if safe.width <= 0 or safe.height <= 0:
        raise _schema_error("safe_area width and height must be positive", "safe_area", "E_SAFE_AREA")
    if safe.right > width or safe.bottom > height:
        raise _schema_error("safe_area must lie inside the frame", "safe_area", "E_SAFE_AREA")

    style = parse_style(obj.get("style"), "style")

    events_raw = obj.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise _schema_error("events must be a non-empty array", "events")
    events = tuple(parse_event(item, safe, i) for i, item in enumerate(events_raw))
    seen: set[str] = set()
    for event in events:
        if event.id in seen:
            raise _schema_error("Duplicate event id", "events.id", "E_EVENT_ID")
        seen.add(event.id)

    edits_raw = obj.get("edits", [])
    if not isinstance(edits_raw, list):
        raise _schema_error("edits must be an array", "edits")
    edits = tuple(parse_edit(item, i) for i, item in enumerate(edits_raw))

    runtime_obj = _required_object(obj.get("runtime", {}), "runtime")
    concurrency = _positive_int(runtime_obj.get("concurrency", 1), "runtime.concurrency", "E_RUNTIME")
    memory_budget = _positive_int(
        runtime_obj.get("memory_budget_bytes", 64 * 1024 * 1024),
        "runtime.memory_budget_bytes",
        "E_RUNTIME",
    )
    block_ms = _positive_int(runtime_obj.get("block_ms", 500), "runtime.block_ms", "E_RUNTIME")
    if block_ms % frame_ms != 0:
        raise _schema_error("block_ms must be an exact multiple of frame_ms", "runtime.block_ms", "E_RUNTIME")
    state_dir = relative_or_absolute(
        Path(str(runtime_obj.get("state_dir", "state"))), source_path.parent
    )
    output_dir = relative_or_absolute(
        Path(str(runtime_obj.get("output_dir", "output"))), source_path.parent
    )
    if state_dir == output_dir:
        raise _schema_error("state_dir and output_dir must be distinct", "runtime", "E_PATH")
    runtime = RuntimeConfig(
        concurrency=concurrency,
        memory_budget_bytes=memory_budget,
        block_ms=block_ms,
        state_dir=state_dir,
        output_dir=output_dir,
    )

    maximum_end = max(event.end_ms for event in events)
    duration = obj.get("duration_ms", maximum_end)
    if not is_int(duration) or duration <= 0:
        raise _schema_error("duration_ms must be a positive integer", "duration_ms", "E_FRAME")
    if duration < maximum_end:
        raise _schema_error("duration_ms must cover all events", "duration_ms", "E_TIME")

    return Spec(
        schema_version=schema_version,
        frame=FrameConfig(width, height, frame_ms, background),
        safe_area=safe,
        style=style,
        events=events,
        edits=edits,
        runtime=runtime,
        duration_ms=duration,
        source_path=source_path,
        raw=copy.deepcopy(obj),
    )


def load_spec(path: Path) -> Spec:
    try:
        data = read_json(path)
    except FileNotFoundError:
        raise EngineError("E_INPUT", "Input file not found", details={"path": str(path)})
    except (OSError, UnicodeDecodeError) as exc:
        raise EngineError(
            "E_INPUT",
            "Unable to read input JSON",
            details={"path": str(path), "reason": str(exc)},
        )
    except ValueError as exc:
        raise EngineError(
            "E_INPUT", "Input is not valid JSON", details={"path": str(path), "reason": str(exc)}
        )
    if not isinstance(data, dict):
        raise _schema_error("Top-level JSON value must be an object", "$")
    return parse_spec(data, path.resolve())
