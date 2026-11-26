"""Deterministic range routing and previous-version drift comparison."""

from __future__ import annotations

import bisect
from typing import Any

from .fixedpoint import event_waterline, quantize_ratio
from .spec import (
    LOWER_EXCLUSIVE,
    LOWER_INCLUSIVE,
    EventSpec,
    RoutingSpec,
    TableSpec,
    ensure_spec_difference,
)
from .errors import spec_error


def _find_interval(table: TableSpec, key: int) -> int | None:
    boundaries = table.boundaries
    if table.boundary_rule == LOWER_INCLUSIVE:
        if key < boundaries[0] or key >= boundaries[-1]:
            return None
        return bisect.bisect_right(boundaries, key) - 1
    if key <= boundaries[0] or key > boundaries[-1]:
        return None
    return bisect.bisect_left(boundaries, key) - 1


def shard_id(table_id: str, interval_index: int) -> str:
    return f"{table_id}:r{interval_index}"


def route_table_key(table: TableSpec, key: int) -> dict[str, Any] | None:
    interval_index = _find_interval(table, key)
    if interval_index is None:
        return None
    lower = table.boundaries[interval_index]
    upper = table.boundaries[interval_index + 1]
    boundary_hit: dict[str, Any] | None = None
    if table.boundary_rule == LOWER_INCLUSIVE and key in table.boundaries:
        boundary_hit = {
            "boundary": key,
            "side": "lower",
            "ownership": "right_interval",
            "rule": table.boundary_rule,
            "interval_index": interval_index,
        }
    elif table.boundary_rule == LOWER_EXCLUSIVE and key in table.boundaries:
        boundary_hit = {
            "boundary": key,
            "side": "upper",
            "ownership": "left_interval",
            "rule": table.boundary_rule,
            "interval_index": interval_index,
        }
    return {
        "table_id": table.id,
        "table_key_column": table.key_column,
        "database_id": table.database_assignment[interval_index],
        "shard": shard_id(table.id, interval_index),
        "interval_index": interval_index,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": upper - lower,
        "route_version": table.route_version,
        "boundary_rule": table.boundary_rule,
        "boundary_hit": boundary_hit,
        "relative_position": {
            "numerator": key - lower,
            "denominator": upper - lower,
            "rounding": "toward_zero",
        },
    }


def route_key(spec: RoutingSpec, table_id: str, key: int) -> dict[str, Any] | None:
    table = spec.table(table_id)
    return route_table_key(table, key)


def _nearest_boundary_distance(old_table: TableSpec | None, new_table: TableSpec, key: int) -> tuple[int | None, int | None]:
    changed = set(new_table.boundaries)
    if old_table is not None:
        changed ^= set(old_table.boundaries)
    if not changed:
        return None, None
    boundary = min(changed, key=lambda candidate: (abs(key - candidate), candidate))
    return boundary, key - boundary


def _drift_direction(old_route: dict[str, Any] | None, new_route: dict[str, Any] | None) -> str:
    if old_route is None and new_route is None:
        return "unroutable"
    if old_route is None:
        return "entered"
    if new_route is None:
        return "exited"
    old_index = old_route["interval_index"]
    new_index = new_route["interval_index"]
    if new_index > old_index:
        return "right"
    if new_index < old_index:
        return "left"
    return "same"


def drift_record(
    spec: RoutingSpec,
    previous: RoutingSpec,
    event: EventSpec,
) -> dict[str, Any]:
    old_table = next((table for table in previous.tables if table.id == event.table), None)
    new_table = spec.table(event.table)
    old_route = route_table_key(old_table, event.key) if old_table is not None else None
    new_route = route_table_key(new_table, event.key)
    direction = _drift_direction(old_route, new_route)
    moved = (
        old_route is None
        or new_route is None
        or old_route["database_id"] != new_route["database_id"]
        or old_route["shard"] != new_route["shard"]
    )
    changed_boundary, signed_distance = _nearest_boundary_distance(old_table, new_table, event.key)
    relative_numerator = new_route["relative_position"]["numerator"] if new_route else 0
    relative_denominator = new_route["relative_position"]["denominator"] if new_route else 1
    fp_offset = quantize_ratio(relative_numerator, relative_denominator, spec.scale_factor)
    return {
        "moved": moved,
        "direction": direction,
        "old_database_id": old_route["database_id"] if old_route else None,
        "old_shard": old_route["shard"] if old_route else None,
        "old_interval_index": old_route["interval_index"] if old_route else None,
        "new_database_id": new_route["database_id"] if new_route else None,
        "new_shard": new_route["shard"] if new_route else None,
        "new_interval_index": new_route["interval_index"] if new_route else None,
        "old_route_version": previous.route_version,
        "new_route_version": spec.route_version,
        "boundary_ownership_rule": new_table.boundary_rule,
        "boundary_hit": new_route["boundary_hit"] if new_route else None,
        "nearest_changed_boundary": changed_boundary,
        "distance_to_changed_boundary": signed_distance,
        "relative_offset": {
            "numerator": relative_numerator,
            "denominator": relative_denominator,
            "scale_factor": spec.scale_factor,
            "fixedpoint": fp_offset,
            "rounding": "toward_zero",
        },
    }


def build_event_conclusion(
    spec: RoutingSpec,
    previous: RoutingSpec | None,
    event: EventSpec,
) -> dict[str, Any]:
    current_route = route_key(spec, event.table, event.key)
    if current_route is None:
        raise spec_error(f"events[seq={event.seq}].key", "event is not routable in the current route table")
    if previous is None:
        old_route = None
        drift = {
            "moved": None,
            "direction": "no_previous_version",
            "old_database_id": None,
            "old_shard": None,
            "old_interval_index": None,
            "new_database_id": current_route["database_id"],
            "new_shard": current_route["shard"],
            "new_interval_index": current_route["interval_index"],
            "old_route_version": None,
            "new_route_version": spec.route_version,
            "boundary_ownership_rule": current_route["boundary_rule"],
            "boundary_hit": current_route["boundary_hit"],
            "nearest_changed_boundary": None,
            "distance_to_changed_boundary": None,
            "relative_offset": {
                "numerator": current_route["relative_position"]["numerator"],
                "denominator": current_route["relative_position"]["denominator"],
                "scale_factor": spec.scale_factor,
                "fixedpoint": quantize_ratio(
                    current_route["relative_position"]["numerator"],
                    current_route["relative_position"]["denominator"],
                    spec.scale_factor,
                ),
                "rounding": "toward_zero",
            },
        }
    else:
        old_table = next((table for table in previous.tables if table.id == event.table), None)
        old_route = route_table_key(old_table, event.key) if old_table is not None else None
        drift = drift_record(spec, previous, event)
    waterline = event_waterline(
        spec.scale_factor,
        current_route["relative_position"]["numerator"],
        current_route["relative_position"]["denominator"],
        current_route["interval_width"],
    )
    return {
        "seq": event.seq,
        "table": event.table,
        "key": event.key,
        "op": event.op,
        "route": current_route,
        "previous_route": old_route,
        "drift": drift,
        "waterline": waterline,
    }


def build_route_report(
    spec: RoutingSpec,
    previous: RoutingSpec | None = None,
    *,
    peak_in_flight: dict[str, int] | None = None,
) -> dict[str, Any]:
    if previous is not None:
        ensure_spec_difference(spec, previous)
    events = [build_event_conclusion(spec, previous, event) for event in spec.events]
    from .fixedpoint import build_waterline

    waterline = build_waterline(spec, events, peak_in_flight=peak_in_flight)
    return {
        "format_version": spec.format_version,
        "route_version": spec.route_version,
        "previous_route_version": previous.route_version if previous is not None else None,
        "spec_digest": spec.digest,
        "previous_spec_digest": previous.digest if previous is not None else None,
        "boundary_ownership": {
            table.id: table.boundary_rule for table in sorted(spec.tables, key=lambda item: item.id)
        },
        "events": events,
        "waterline": waterline,
    }


__all__ = [
    "build_event_conclusion",
    "build_route_report",
    "drift_record",
    "route_key",
    "route_table_key",
    "shard_id",
]
