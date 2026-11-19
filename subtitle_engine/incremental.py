"""Sequential edit application with dependency-based invalidation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import EngineError
from .layout import EventLayout, build_event_layout
from .spec import STYLE_FIELDS, Edit, Event, Spec, Style, parse_style


@dataclass(frozen=True)
class EditStep:
    index: int
    op: str
    event_id: str | None
    field: str | None
    affected_event_ids: tuple[str, ...]
    dirty_blocks: tuple[int, ...]
    dirty_regions: tuple[list[int], ...]
    warning_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.index + 1,
            "op": self.op,
            "event_id": self.event_id,
            "field": self.field,
            "affected_event_ids": list(self.affected_event_ids),
            "dirty_blocks": list(self.dirty_blocks),
            "dirty_regions": [list(region) for region in self.dirty_regions],
            "warning_codes": list(self.warning_codes),
        }


@dataclass(frozen=True)
class IncrementalPlan:
    final_events: tuple[Event, ...]
    final_style: Style
    final_safe_area: Any
    final_duration_ms: int
    layouts: dict[str, EventLayout]
    steps: tuple[EditStep, ...]
    invalidated_blocks: tuple[int, ...]
    event_invalidated_blocks: dict[str, tuple[int, ...]]


def _style_dict(style: Style) -> dict[str, Any]:
    return {
        "font_size": style.font_size,
        "line_height": style.line_height,
        "max_lines": style.max_lines,
        "padding": style.padding,
        "contrast_enhance": style.contrast_enhance,
        "mask_opacity": style.mask_opacity,
    }


def _validate_style_field(style: Style, field_name: str, value: Any) -> Style:
    candidate = _style_dict(style)
    candidate[field_name] = value
    return parse_style(candidate)


def _validate_edited_event(event: Event) -> Event:
    if event.start_ms < 0 or event.end_ms <= event.start_ms:
        raise EngineError(
            "E_TIME",
            "Edited event has an invalid time window",
            details={"event_id": event.id, "start_ms": event.start_ms, "end_ms": event.end_ms},
        )
    if not event.text.strip():
        raise EngineError(
            "E_TEXT", "Edited event text must not be empty", details={"event_id": event.id}
        )
    return event


def _block_set(
    layout: EventLayout, frame_ms: int, frames_per_block: int
) -> set[int]:
    return layout.visible_block_indices(frame_ms, frames_per_block)


def build_incremental_plan(spec: Spec) -> IncrementalPlan:
    """Apply edits one at a time while recomputing only dependent event layouts.

    A frame block is dirty when an affected subtitle is visible in it in either
    the old or new state.  Style fields affect every subtitle; text/time edits
    affect one subtitle.  Blocks not in the union are reusable.
    """

    events: dict[str, Event] = {event.id: event for event in spec.events}
    event_order = [event.id for event in spec.events]
    style = spec.style
    layouts = {
        event.id: build_event_layout(event, style, spec.safe_area)
        for event in spec.events
    }
    accumulated: set[int] = set()
    per_event: dict[str, set[int]] = {event_id: set() for event_id in event_order}
    steps: list[EditStep] = []

    for edit in spec.edits:
        affected_ids: list[str]
        changed = True
        warning_codes: list[str] = []

        if edit.op == "set_style":
            assert edit.field is not None and edit.field in STYLE_FIELDS
            new_style = _validate_style_field(style, edit.field, edit.value)
            changed = new_style != style
            affected_ids = list(event_order)
            style = new_style
        elif edit.op == "set_text":
            event_id = edit.event_id or ""
            old_event = events.get(event_id)
            if old_event is None:
                raise EngineError(
                    "E_EDIT", "Edit references an unknown event", details={"event_id": event_id}
                )
            new_event = _validate_edited_event(old_event.with_text(str(edit.value)))
            changed = new_event.text != old_event.text
            events[event_id] = new_event
            affected_ids = [event_id]
        elif edit.op == "set_time":
            event_id = edit.event_id or ""
            old_event = events.get(event_id)
            if old_event is None:
                raise EngineError(
                    "E_EDIT", "Edit references an unknown event", details={"event_id": event_id}
                )
            assert edit.start_ms is not None and edit.end_ms is not None
            new_event = _validate_edited_event(
                old_event.with_time(edit.start_ms, edit.end_ms)
            )
            changed = (
                new_event.start_ms != old_event.start_ms
                or new_event.end_ms != old_event.end_ms
            )
            events[event_id] = new_event
            affected_ids = [event_id]
        else:  # parse_spec guarantees this cannot occur
            raise EngineError("E_EDIT", "Unknown edit operation")

        old_layouts = {event_id: layouts[event_id] for event_id in affected_ids}
        if changed:
            try:
                new_layouts = {
                    event_id: build_event_layout(events[event_id], style, spec.safe_area)
                    for event_id in affected_ids
                }
            except EngineError as exc:
                details = dict(exc.details)
                details["edit_index"] = edit.index
                raise EngineError(exc.code, exc.message, details=details) from exc
        else:
            new_layouts = old_layouts

        dirty: set[int] = set()
        regions: set[tuple[int, int, int, int]] = set()
        if changed:
            for event_id in affected_ids:
                old = old_layouts[event_id]
                new = new_layouts[event_id]
                event_dirty = _block_set(old, spec.frame.frame_ms, spec.frames_per_block)
                event_dirty |= _block_set(new, spec.frame.frame_ms, spec.frames_per_block)
                dirty |= event_dirty
                per_event[event_id] |= event_dirty
                regions.add(tuple(old.bbox.as_list()))
                regions.add(tuple(new.bbox.as_list()))
                if new.speed_limited:
                    warning_codes.append("E_SPEED_LIMIT")
        layouts.update(new_layouts)
        accumulated |= dirty
        steps.append(
            EditStep(
                index=edit.index,
                op=edit.op,
                event_id=edit.event_id,
                field=edit.field,
                affected_event_ids=tuple(affected_ids),
                dirty_blocks=tuple(sorted(dirty)),
                dirty_regions=tuple(sorted(regions)),
                warning_codes=tuple(sorted(set(warning_codes))),
            )
        )

    final_events = tuple(events[event_id] for event_id in event_order)
    final_duration = max(spec.duration_ms, max(event.end_ms for event in final_events))
    return IncrementalPlan(
        final_events=final_events,
        final_style=style,
        final_safe_area=spec.safe_area,
        final_duration_ms=final_duration,
        layouts=layouts,
        steps=tuple(steps),
        invalidated_blocks=tuple(sorted(accumulated)),
        event_invalidated_blocks={
            event_id: tuple(sorted(blocks)) for event_id, blocks in per_event.items()
        },
    )


def materialize_final_spec(spec: Spec, plan: IncrementalPlan) -> Spec:
    return replace(
        spec,
        events=plan.final_events,
        style=plan.final_style,
        duration_ms=plan.final_duration_ms,
    )
