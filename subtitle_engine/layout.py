"""Event layout, reading-rate checks, and style dependency records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import EngineError
from .metrics import line_pixel_height, wrap_text
from .spec import Event, Rect, Spec, Style


@dataclass(frozen=True)
class EventLayout:
    event: Event
    lines: tuple[str, ...]
    line_widths: tuple[int, ...]
    bbox: Rect
    reading_chars: int
    min_display_ms: int
    speed_limited: bool
    duration_ms: int
    wrap_width: int
    style: Style
    safe_area: Rect

    def visible_frame_range(self, frame_ms: int) -> tuple[int, int]:
        start = math.ceil(self.event.start_ms / frame_ms)
        end = math.ceil(self.event.end_ms / frame_ms)
        return start, end

    def visible_block_indices(self, frame_ms: int, frames_per_block: int) -> set[int]:
        start, end = self.visible_frame_range(frame_ms)
        if end <= start:
            return set()
        return set(range(start // frames_per_block, (end - 1) // frames_per_block + 1))

    def dependencies(self) -> dict[str, Any]:
        return {
            "font_size": self.style.font_size,
            "line_height": self.style.line_height,
            "max_lines": self.style.max_lines,
            "padding": self.style.padding,
            "contrast_enhance": self.style.contrast_enhance,
            "mask_opacity": self.style.mask_opacity,
            "safe_area": self.safe_area.as_list(),
            "anchor": [
                self.event.anchor.x,
                self.event.anchor.y,
                self.event.anchor.horizontal,
                self.event.anchor.vertical,
            ],
        }

    def signature(self) -> tuple[Any, ...]:
        return (
            self.event.text,
            self.event.start_ms,
            self.event.end_ms,
            tuple(self.lines),
            tuple(self.line_widths),
            self.bbox.as_list(),
            self.min_display_ms,
            self.speed_limited,
            self.style,
            self.safe_area,
        )


def reading_character_count(text: str) -> int:
    return len(text)


def build_event_layout(event: Event, style: Style, safe: Rect) -> EventLayout:
    lines, widths = wrap_text(event.text, style, safe.width)
    if len(lines) > style.max_lines:
        raise EngineError(
            "E_LINE_OVERFLOW",
            "Text requires more lines than max_lines; refusing to truncate",
            details={
                "event_id": event.id,
                "line_count": len(lines),
                "max_lines": style.max_lines,
                "lines": lines,
            },
        )

    line_height = line_pixel_height(style)
    box_width = max(1, max(widths, default=0)) + int(math.ceil(2.0 * style.padding))
    box_height = max(1, line_height * len(lines)) + int(math.ceil(2.0 * style.padding))

    anchor = event.anchor
    if anchor.horizontal == "left":
        x = anchor.x
    elif anchor.horizontal == "right":
        x = anchor.x - box_width
    else:
        x = anchor.x - box_width / 2.0

    if anchor.vertical == "top":
        y = anchor.y
    elif anchor.vertical == "bottom":
        y = anchor.y - box_height
    else:
        y = anchor.y - box_height / 2.0

    x_int = int(math.floor(x + 0.5))
    y_int = int(math.floor(y + 0.5))
    if (
        x_int < safe.x
        or y_int < safe.y
        or x_int + box_width > safe.right
        or y_int + box_height > safe.bottom
    ):
        raise EngineError(
            "E_LAYOUT",
            "Subtitle bounding box does not fit inside the safe area",
            details={
                "event_id": event.id,
                "bbox": [x_int, y_int, box_width, box_height],
                "safe_area": safe.as_list(),
            },
        )

    chars = reading_character_count(event.text)
    required_ms = int(math.ceil(chars * 1000.0 / event.max_chars_per_sec))
    duration = event.end_ms - event.start_ms
    return EventLayout(
        event=event,
        lines=tuple(lines),
        line_widths=tuple(widths),
        bbox=Rect(x_int, y_int, box_width, box_height),
        reading_chars=chars,
        min_display_ms=required_ms,
        speed_limited=duration < required_ms,
        duration_ms=duration,
        wrap_width=available_width(style, safe),
        style=style,
        safe_area=safe,
    )


def available_width(style: Style, safe: Rect) -> int:
    return int(math.floor(safe.width - 2.0 * style.padding))


def all_layouts(spec: Spec) -> dict[str, EventLayout]:
    return {
        event.id: build_event_layout(event, spec.style, spec.safe_area)
        for event in spec.events
    }
