"""Strict NDJSON observation loading and state evolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .canonical import parse_json_strict
from .declarations import validate_labels
from .errors import event_error
from .models import FIELD_ORDER, ObservationEvent, TargetSet


EVENT_KEYS = {"seq", "workload", "field", "value", "at_ms"}


def _validate_observed_value(field: str, value: Any, *, item: Any) -> None:
    if field in {"replicas", "resource_limit", "probe_ms"}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise event_error(
                f"observed value for {field} must be an integer",
                item=item,
                field=field,
            )
        return
    if field == "image":
        if not isinstance(value, str) or not value:
            raise event_error("observed image must be a non-empty string", item=item, field=field)
        return
    if field == "labels":
        error = validate_labels(value)
        if error:
            raise event_error(error, item=item, field=field)
        return
    if field == "parent_ref":
        if value is not None and (not isinstance(value, str) or not value):
            raise event_error("observed parent_ref must be null or a non-empty string", item=item, field=field)


def validate_events(raw_events: Iterable[dict[str, Any]], target: TargetSet) -> list[ObservationEvent]:
    collected: list[tuple[int, ObservationEvent]] = []
    seen_sequences: dict[int, int] = {}

    for line_number, raw in enumerate(raw_events, start=1):
        item = {"line": line_number, "seq": raw.get("seq") if isinstance(raw, dict) else None}
        if not isinstance(raw, dict):
            raise event_error("event must be a JSON object", item=item, field="<event>")
        for key in raw:
            if key not in EVENT_KEYS:
                raise event_error(f"unknown event field: {key}", item=item, field=key)
        for key in ("seq", "workload", "field", "value", "at_ms"):
            if key not in raw:
                raise event_error("event field is missing", item=item, field=key)

        seq = raw["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise event_error("seq must be an integer", item=item, field="seq")
        if seq in seen_sequences:
            raise event_error(
                "event sequence must be unique",
                item=item,
                field="seq",
                details={"seq": seq, "first_line": seen_sequences[seq]},
            )
        seen_sequences[seq] = line_number

        workload = raw["workload"]
        if not isinstance(workload, str) or not workload:
            raise event_error("workload must be a non-empty string", item=item, field="workload")
        if workload not in target.workloads:
            raise event_error(
                f"unknown workload: {workload}",
                item=item,
                field="workload",
                details={"known_workloads": sorted(target.workloads)},
            )

        field = raw["field"]
        if not isinstance(field, str) or field not in FIELD_ORDER:
            raise event_error(
                f"unknown field path: {field!r}",
                item=item,
                field="field",
                details={"known_fields": list(FIELD_ORDER)},
            )

        _validate_observed_value(field, raw["value"], item=item)

        at_ms = raw["at_ms"]
        if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
            raise event_error("at_ms must be a non-negative integer", item=item, field="at_ms")
        collected.append(
            (
                line_number,
                ObservationEvent(
                    seq=seq,
                    workload=workload,
                    field=field,
                    value=deepcopy(raw["value"]),
                    at_ms=at_ms,
                ),
            )
        )

    collected.sort(key=lambda item: item[1].seq)
    events = [event for _, event in collected]
    for expected, event in enumerate(events, start=1):
        if event.seq != expected:
            raise event_error(
                f"event sequence must be contiguous; expected {expected}",
                item={"seq": event.seq},
                field="seq",
                details={"expected": expected, "actual": event.seq},
            )
    previous_at: int | None = None
    for event in events:
        if previous_at is not None and event.at_ms < previous_at:
            raise event_error(
                "at_ms must be monotonic non-decreasing when events are ordered by seq",
                item={"seq": event.seq},
                field="at_ms",
                details={"previous": previous_at, "actual": event.at_ms},
            )
        previous_at = event.at_ms
    return events


def load_events(path: str | Path, target: TargetSet) -> list[ObservationEvent]:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise event_error(f"cannot read events file: {exc}", item=str(file_path), field="<file>") from exc

    raw_events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw = parse_json_strict(line, source=f"{file_path}:{line_number}")
        if not isinstance(raw, dict):
            raise event_error("event line must be a JSON object", item={"line": line_number}, field="<event>")
        raw_events.append(raw)
    return validate_events(raw_events, target)


def apply_event(actual_state: dict[str, dict[str, Any]], event: ObservationEvent) -> None:
    actual_state.setdefault(event.workload, {})[event.field] = deepcopy(event.value)


def apply_events(
    actual_state: dict[str, dict[str, Any]],
    events: Iterable[ObservationEvent],
) -> dict[str, dict[str, Any]]:
    for event in events:
        apply_event(actual_state, event)
    return actual_state
