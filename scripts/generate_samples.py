#!/usr/bin/env python3
"""Generate deterministic local fixtures for the pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SEED = 173


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def generate_events() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    event_types = [
        "view",
        "click",
        "add_to_cart",
        "view",
        "purchase",
        "click",
        "view",
        "add_to_cart",
        "click",
        "purchase",
        "view",
        "click",
        "purchase",
        "view",
        "add_to_cart",
        "purchase",
        "click",
        "view",
        "click",
        "purchase",
        "view",
        "add_to_cart",
        "click",
        "purchase",
    ]
    users = ["u1", "u2", "u1", "u3", "u2", "u1", "u3", "u2", "u1", "u2"]
    items = ["i1", "i2", "i1", "i3", "i2", "i4", "i1", "i3", "i2", "i4"]
    events: list[dict[str, Any]] = []
    for index, event_type in enumerate(event_types, 1):
        user_id = users[(index - 1) % len(users)]
        item_id = items[(index - 1) % len(items)]
        price = round(28 + (index * 7) % 67 + rng.choice([0.0, 0.25, 0.5, 0.75]), 2)
        events.append(
            {
                "event_id": f"e{index:04d}",
                "user_id": user_id,
                "item_id": item_id,
                "event_type": event_type,
                "timestamp": index * 1000,
                "context": {
                    "price": price,
                    "dwell_ms": 300 + ((index * 137) % 1900),
                    "rating": round(3.0 + ((index * 13) % 20) / 10.0, 1),
                },
            }
        )
    return events


def base_features() -> dict[str, Any]:
    return {
        "features": [
            {
                "name": "user_purchase_count_10s",
                "entity": "user",
                "window_ms": 10000,
                "aggregation": "count",
                "empty_value": 0,
                "filter": {"field": "event_type", "equals": "purchase"},
                "upstream_features": [],
            },
            {
                "name": "user_purchase_value_10s",
                "entity": "user",
                "window_ms": 10000,
                "aggregation": "sum",
                "source_field": "context.price",
                "empty_value": 0,
                "filter": {"field": "event_type", "equals": "purchase"},
                "upstream_features": [],
            },
            {
                "name": "user_view_count_5s",
                "entity": "user",
                "window_ms": 5000,
                "aggregation": "count",
                "empty_value": 0,
                "filter": {"field": "event_type", "equals": "view"},
                "upstream_features": [],
            },
            {
                "name": "item_avg_price_10s",
                "entity": "item",
                "window_ms": 10000,
                "aggregation": "mean",
                "source_field": "context.price",
                "empty_value": 0,
                "upstream_features": [],
            },
            {
                "name": "item_view_count_5s",
                "entity": "item",
                "window_ms": 5000,
                "aggregation": "count",
                "empty_value": 0,
                "filter": {"field": "event_type", "equals": "view"},
                "upstream_features": [],
            },
            {
                "name": "item_latest_dwell_8s",
                "entity": "item",
                "window_ms": 8000,
                "aggregation": "latest",
                "source_field": "context.dwell_ms",
                "empty_value": -1,
                "upstream_features": [],
            },
            {
                "name": "user_purchase_value_smooth_12s",
                "entity": "user",
                "window_ms": 12000,
                "aggregation": "mean",
                "source_feature": "user_purchase_value_10s",
                "empty_value": 0,
                "upstream_features": ["user_purchase_value_10s"],
            },
            {
                "name": "item_avg_price_smooth_15s",
                "entity": "item",
                "window_ms": 15000,
                "aggregation": "mean",
                "source_feature": "item_avg_price_10s",
                "empty_value": 0,
                "upstream_features": ["item_avg_price_10s"],
            },
            {
                "name": "item_price_chain_20s",
                "entity": "item",
                "window_ms": 20000,
                "aggregation": "mean",
                "source_feature": "item_avg_price_smooth_15s",
                "empty_value": 0,
                "upstream_features": ["item_avg_price_smooth_15s"],
            },
        ]
    }


def final_features() -> dict[str, Any]:
    features = base_features()
    by_name = {feature["name"]: feature for feature in features["features"]}
    by_name["item_view_count_5s"]["window_ms"] = 9000
    by_name["item_latest_dwell_8s"]["aggregation"] = "mean"
    return features


def window_changed_features() -> dict[str, Any]:
    features = base_features()
    by_name = {feature["name"]: feature for feature in features["features"]}
    by_name["item_view_count_5s"]["window_ms"] = 9000
    return features


def base_labels() -> dict[str, Any]:
    return {
        "name": "purchase_quality",
        "trigger_event_types": ["purchase"],
        "delay_ms": 3000,
        "feature_names": [
            "user_purchase_count_10s",
            "user_purchase_value_10s",
            "item_avg_price_10s",
        ],
        "positive_rule": {
            "all": [
                {"feature": "user_purchase_count_10s", "operator": "ge", "value": 1},
                {"feature": "item_avg_price_10s", "operator": "ge", "value": 60},
            ]
        },
        "negative_rule": {
            "any": [
                {"feature": "user_purchase_count_10s", "operator": "lt", "value": 1},
                {"feature": "item_avg_price_10s", "operator": "lt", "value": 60},
            ]
        },
    }


def final_labels() -> dict[str, Any]:
    labels = base_labels()
    labels["delay_ms"] = 5000
    return labels


def generate(output: Path) -> None:
    events = generate_events()
    final_events = json.loads(json.dumps(events))
    backfilled = next(event for event in final_events if event["event_id"] == "e0010")
    backfilled["context"]["price"] = round(backfilled["context"]["price"] + 25.0, 2)
    backfilled["context"]["dwell_ms"] += 500

    config = {
        "allowed_event_types": ["view", "click", "add_to_cart", "purchase"],
        "max_workers": 4,
        "stage_delay_ms": 0,
    }
    serial_config = dict(config)
    serial_config["max_workers"] = 1

    base = base_features()
    window_changed = window_changed_features()
    final = final_features()
    labels_base = base_labels()
    labels_final = final_labels()
    edits = {
        "edits": [
            {
                "kind": "feature_window",
                "feature": "item_view_count_5s",
                "window_ms": 9000,
            },
            {
                "kind": "feature_aggregation",
                "feature": "item_latest_dwell_8s",
                "aggregation": "mean",
            },
            {"kind": "label_delay", "delay_ms": 5000},
            {
                "kind": "backfill",
                "event_id": "e0010",
                "timestamp": 10000,
                "patch": {"context.price": backfilled["context"]["price"], "context.dwell_ms": backfilled["context"]["dwell_ms"]},
            },
        ]
    }
    injection_edits = {
        "edits": [
            {
                "kind": "backfill",
                "event_id": "e0020",
                "timestamp": 20000,
                "patch": {"context.price": 88.0},
            }
        ]
    }
    injections = {
        "injections": [
            {
                "id": "INJ-01",
                "event_id": "e0004",
                "stage": "parse",
                "kind": "non_finite_context",
                "field": "context.price",
            },
            {
                "id": "INJ-02",
                "event_id": "e0007",
                "stage": "parse",
                "kind": "unknown_event_type",
                "value": "teleport",
            },
            {
                "id": "INJ-03",
                "event_id": "e0012",
                "stage": "parse",
                "kind": "unknown_event_field",
                "field": "injected_marker",
                "value": True,
            },
            {
                "id": "INJ-04",
                "stage": "label",
                "kind": "missing_label_feature",
                "feature": "ghost_feature",
            },
            {
                "id": "INJ-05",
                "event_id": "e0020",
                "stage": "backfill",
                "kind": "backfill_before_cursor",
            },
        ]
    }
    manifest = {
        "seed": SEED,
        "event_count": len(events),
        "base_event_count": 10,
        "backfill_event_id": "e0010",
        "sequence": [
            "reset with base events/features/labels",
            "append pre-backfill events",
            "change item_view_count_5s window to 9000",
            "switch item_latest_dwell_8s from latest to mean",
            "adjust label delay to 5000",
            "backfill e0010 context",
        ],
    }

    write_jsonl(output / "events_base_10.jsonl", events[:10])
    write_jsonl(output / "events_pre_backfill.jsonl", events)
    write_jsonl(output / "events.jsonl", final_events)
    write_json(output / "features_base.json", base)
    write_json(output / "features_window_changed.json", window_changed)
    write_json(output / "features.json", final)
    write_json(output / "labels_base.json", labels_base)
    write_json(output / "labels.json", labels_final)
    write_json(output / "config.json", config)
    write_json(output / "config_serial.json", serial_config)
    write_json(output / "edits.json", edits)
    write_json(output / "edits_injection.json", injection_edits)
    write_json(output / "injections.json", injections)
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("samples"))
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()

