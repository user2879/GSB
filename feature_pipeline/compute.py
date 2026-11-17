"""Deterministic windowed feature computation and orchestration."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import deep_get, is_finite_number, sha256_value
from .models import FeatureDefinition, FeatureTable
from .validation import E_FEATURE_COMPUTE, failure


class FeatureComputeError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureTask:
    name: str
    position: int
    anchor_event_ids: tuple[str, ...] | None
    recompute_all: bool

    @property
    def stable_key(self) -> tuple[int, str]:
        return (self.position, self.name)


def entity_id(definition: FeatureDefinition, event: dict[str, Any]) -> str:
    return event["user_id"] if definition.entity == "user" else event["item_id"]


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record["feature"], record["event_id"])


def sort_feature_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            record["feature"],
            record["event_timestamp"],
            record["event_id"],
            record["entity_type"],
            record["entity_id"],
        ),
    )


def compute_feature_records(
    definition: FeatureDefinition,
    events: list[dict[str, Any]],
    feature_lookup: dict[tuple[str, str], dict[str, Any]],
    anchor_event_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute records for one feature.

    ``anchor_event_ids=None`` computes every event matching the feature entity.
    A supplied set restricts the task to newly needed anchors while old values
    remain available in the caller's cache.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        key = entity_id(definition, event)
        grouped.setdefault(key, []).append(event)
    for entity_events in grouped.values():
        entity_events.sort(key=lambda event: (event["timestamp"], event["event_id"]))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entity, entity_events in sorted(grouped.items()):
        timestamps = [event["timestamp"] for event in entity_events]
        for index, anchor in enumerate(entity_events):
            if anchor_event_ids is not None and anchor["event_id"] not in anchor_event_ids:
                continue
            cutoff = anchor["timestamp"] - definition.window_ms
            start = _lower_bound(timestamps, cutoff)
            candidates = entity_events[start : index + 1]
            try:
                record = _compute_one(definition, entity, anchor, candidates, feature_lookup)
            except Exception as exc:  # noqa: BLE001 - isolate one anchor's data fault
                failures.append(
                    failure(
                        E_FEATURE_COMPUTE,
                        "feature",
                        anchor["event_id"],
                        type(exc).__name__,
                        str(exc),
                        feature=definition.name,
                    )
                )
                continue
            records.append(record)
    return sort_feature_records(records), failures


def _compute_one(
    definition: FeatureDefinition,
    entity: str,
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    values: list[int | float] = []
    for candidate in candidates:
        if definition.event_filter is not None:
            try:
                filtered_value = deep_get(candidate, definition.event_filter.field)
            except KeyError:
                continue
            if filtered_value != definition.event_filter.equals:
                continue
        if definition.source_feature is not None:
            upstream = feature_lookup.get((definition.source_feature, candidate["event_id"]))
            if upstream is None:
                raise FeatureComputeError(
                    f"missing upstream {definition.source_feature} at event {candidate['event_id']}"
                )
            source_value = upstream["value"]
            upstream_key = upstream["cache_key"]
        elif definition.aggregation == "count":
            source_value = 1
            upstream_key = None
        else:
            try:
                source_value = deep_get(candidate, definition.source_field or "")
            except KeyError as exc:
                raise FeatureComputeError(
                    f"missing source field {definition.source_field} at event {candidate['event_id']}"
                ) from exc
            upstream_key = None
        if not is_finite_number(source_value):
            raise FeatureComputeError(
                f"source value {definition.source_field or definition.source_feature} is not finite numeric"
            )
        values.append(source_value)
        input_record: dict[str, Any] = {
            "event_id": candidate["event_id"],
            "event_fingerprint": feature_event_fingerprint(definition, candidate),
        }
        if upstream_key is not None:
            input_record["upstream_cache_key"] = upstream_key
        inputs.append(input_record)

    empty = not inputs
    if empty:
        value: int | float = definition.empty_value
    elif definition.aggregation == "count":
        value = len(values)
    elif definition.aggregation == "sum":
        value = math.fsum(float(item) for item in values)
    elif definition.aggregation == "mean":
        value = math.fsum(float(item) for item in values) / len(values)
    elif definition.aggregation == "latest":
        value = values[-1]
    else:
        raise FeatureComputeError(f"unsupported aggregation: {definition.aggregation}")
    if not is_finite_number(value):
        raise FeatureComputeError("aggregate value is not finite")

    cache_payload = {
        "schema": "feature-cache-v1",
        "feature": definition.name,
        "entity_type": definition.entity,
        "entity_id": entity,
        "anchor_event_id": anchor["event_id"],
        "window": {
            "window_ms": definition.window_ms,
            "start_inclusive": anchor["timestamp"] - definition.window_ms,
            "end_inclusive": anchor["timestamp"],
        },
        "aggregation_fingerprint": definition.aggregation_fingerprint,
        "definition_fingerprint": definition.fingerprint,
        "inputs": inputs,
        "empty": empty,
    }
    return {
        "feature": definition.name,
        "entity_type": definition.entity,
        "entity_id": entity,
        "event_id": anchor["event_id"],
        "event_timestamp": anchor["timestamp"],
        "value": value,
        "empty": empty,
        "definition_hash": definition.fingerprint,
        "cache_key": sha256_value(cache_payload),
    }


def compute_feature_levels(
    table: FeatureTable,
    events: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    invalidated: set[str],
    *,
    max_workers: int,
    force_missing_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Recompute invalid features and append missing records.

    Returns ``(records, computed_names, failures)``. Features are scheduled in
    topological levels. Tasks inside a level are submitted concurrently and
    consumed in ``(topological_position, name)`` order, so scheduling cannot
    affect results or lists.
    """

    force_missing_names = force_missing_names or set()
    existing: dict[tuple[str, str], dict[str, Any]] = {
        record_key(record): record
        for record in previous_records
        if record["feature"] in table.definitions
    }
    current_records: list[dict[str, Any]] = [
        record for record in previous_records if record["feature"] in table.definitions
    ]
    all_event_ids = {event["event_id"] for event in events}
    computed_names: list[str] = []
    failures: list[dict[str, Any]] = []

    for level in table.levels:
        tasks: list[FeatureTask] = []
        for name in level:
            definition = table.definitions[name]
            old_keys = {event_id for feature, event_id in existing if feature == name}
            missing_ids = all_event_ids - old_keys
            recompute_all = name in invalidated or name in force_missing_names or not old_keys
            if recompute_all:
                tasks.append(
                    FeatureTask(
                        name=name,
                        position=table.positions[name],
                        anchor_event_ids=None,
                        recompute_all=True,
                    )
                )
            elif missing_ids:
                tasks.append(
                    FeatureTask(
                        name=name,
                        position=table.positions[name],
                        anchor_event_ids=tuple(sorted(missing_ids)),
                        recompute_all=False,
                    )
                )
        tasks.sort(key=lambda task: task.stable_key)
        if not tasks:
            continue

        lookup_snapshot = dict(existing)

        def execute(task: FeatureTask) -> tuple[FeatureTask, list[dict[str, Any]], list[dict[str, Any]]]:
            definition = table.definitions[task.name]
            anchor_ids = None if task.anchor_event_ids is None else set(task.anchor_event_ids)
            records, task_failures = compute_feature_records(
                definition, events, lookup_snapshot, anchor_ids
            )
            return task, records, task_failures

        results: list[tuple[FeatureTask, list[dict[str, Any]], list[dict[str, Any]]]] = []
        worker_count = max(1, min(max_workers, len(tasks)))
        if worker_count == 1:
            results = [execute(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="feature") as pool:
                futures = [pool.submit(execute, task) for task in tasks]
                results = [future.result() for future in futures]
        results.sort(key=lambda item: item[0].stable_key)

        for task, new_records, task_failures in results:
            if task.recompute_all:
                current_records = [
                    record for record in current_records if record["feature"] != task.name
                ]
                existing = {
                    key: value for key, value in existing.items() if key[0] != task.name
                }
            for record in new_records:
                existing[record_key(record)] = record
                current_records.append(record)
            for task_failure in task_failures:
                task_failure["feature"] = task.name
                failures.append(task_failure)
            computed_names.append(task.name)

    return sort_feature_records(current_records), sorted(computed_names), failures


def _lower_bound(values: list[int], target: int) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def feature_event_fingerprint(
    definition: FeatureDefinition, event: dict[str, Any]
) -> str:
    """Fingerprint only the event facets that can affect this feature.

    This keeps a backfill of an unrelated context field from invalidating a
    feature that never reads it, while still making every contributing event
    content-addressed in the cache key.
    """

    entity_field = "user_id" if definition.entity == "user" else "item_id"
    payload: dict[str, Any] = {
        "event_id": event["event_id"],
        "timestamp": event["timestamp"],
        entity_field: event[entity_field],
    }
    if definition.source_field is not None:
        try:
            payload["source_field"] = definition.source_field
            payload["source_value"] = deep_get(event, definition.source_field)
        except KeyError:
            payload["source_missing"] = definition.source_field
    if definition.event_filter is not None:
        try:
            payload["filter_field"] = definition.event_filter.field
            payload["filter_value"] = deep_get(event, definition.event_filter.field)
        except KeyError:
            payload["filter_missing"] = definition.event_filter.field
    if definition.source_feature is not None:
        payload["upstream_event_variant"] = (
            event["user_id"] if definition.entity == "user" else event["item_id"]
        )
    return sha256_value(payload)
