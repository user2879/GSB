"""Label-delayed sample materialization from anchor feature snapshots."""

from __future__ import annotations

from typing import Any

from .canonical import sha256_value
from .models import LabelDefinition
from .validation import (
    E_FEATURE_COMPUTE,
    E_LABEL_UNDEFINED,
    apply_rule,
    failure,
)


def sort_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        samples,
        key=lambda sample: (
            sample["generation_timestamp"],
            sample["trigger_event_id"],
            sample["sample_id"],
        ),
    )


def generate_samples(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    label: LabelDefinition,
    cursor: int,
    previous_samples: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[str], list[str], list[str], list[dict[str, Any]]
]:
    """Return samples, regenerated ids, reused ids, invalidated ids, failures."""

    feature_lookup = {(record["feature"], record["event_id"]): record for record in features}
    old_by_id = {sample["sample_id"]: sample for sample in previous_samples}
    candidates: list[dict[str, Any]] = []
    regenerated: list[str] = []
    reused: list[str] = []
    failures: list[dict[str, Any]] = []
    trigger_types = set(label.trigger_event_types)
    feature_names = tuple(sorted(label.feature_names))

    for event in events:
        if trigger_types and event["event_type"] not in trigger_types:
            continue
        anchor_timestamp = event["timestamp"]
        generation_timestamp = anchor_timestamp + label.delay_ms
        if generation_timestamp > cursor:
            continue
        values: dict[str, int | float] = {}
        empty_flags: dict[str, bool] = {}
        cache_keys: dict[str, str] = {}
        missing = False
        for feature_name in feature_names:
            record = feature_lookup.get((feature_name, event["event_id"]))
            if record is None:
                failures.append(
                    failure(
                        E_FEATURE_COMPUTE,
                        "sample",
                        event["event_id"],
                        "MissingFeatureSnapshotError",
                        f"missing {feature_name} snapshot at anchor",
                        feature=feature_name,
                    )
                )
                missing = True
                break
            values[feature_name] = record["value"]
            empty_flags[feature_name] = record["empty"]
            cache_keys[feature_name] = record["cache_key"]
        if missing:
            continue
        try:
            positive = apply_rule(label.positive_rule, values)
            negative = False if positive else apply_rule(label.negative_rule, values)
        except Exception as exc:  # noqa: BLE001 - a bad rule is isolated per anchor
            failures.append(
                failure(
                    E_LABEL_UNDEFINED,
                    "sample",
                    event["event_id"],
                    type(exc).__name__,
                    f"label rule evaluation failed: {exc}",
                    label=label.name,
                )
            )
            continue
        if not positive and not negative:
            failures.append(
                failure(
                    E_LABEL_UNDEFINED,
                    "sample",
                    event["event_id"],
                    "UndefinedLabelError",
                    "neither positive nor negative rule matched",
                    label=label.name,
                )
            )
            continue
        label_value = 1 if positive else 0
        snapshot_version = sha256_value(
            {
                "kind": "sample-feature-snapshot-v1",
                "anchor_event_id": event["event_id"],
                "anchor_timestamp": anchor_timestamp,
                "features": {
                    name: {
                        "value": values[name],
                        "empty": empty_flags[name],
                        "cache_key": cache_keys[name],
                    }
                    for name in feature_names
                },
            }
        )
        sample = {
            "sample_id": f"sample:{label.name}:{event['event_id']}",
            "label_name": label.name,
            "trigger_event_id": event["event_id"],
            "user_id": event["user_id"],
            "item_id": event["item_id"],
            "anchor_timestamp": anchor_timestamp,
            "generation_timestamp": generation_timestamp,
            "label": label_value,
            "features": values,
            "feature_empty": empty_flags,
            "feature_cache_keys": cache_keys,
            "feature_snapshot_version": snapshot_version,
        }
        candidates.append(sample)

    candidates = sort_samples(candidates)
    candidate_ids = {sample["sample_id"] for sample in candidates}
    regenerated = [
        sample["sample_id"]
        for sample in candidates
        if old_by_id.get(sample["sample_id"]) != sample
    ]
    reused = [
        sample["sample_id"]
        for sample in candidates
        if old_by_id.get(sample["sample_id"]) == sample
    ]
    invalidated = sorted(
        sample_id
        for sample_id in old_by_id
        if sample_id not in candidate_ids or old_by_id[sample_id] != _candidate_by_id(candidates, sample_id)
    )
    return candidates, sorted(regenerated), sorted(reused), invalidated, failures


def _candidate_by_id(candidates: list[dict[str, Any]], sample_id: str) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate["sample_id"] == sample_id:
            return candidate
    return None
