from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feature_pipeline.validation import (
    E_EVENT_INVALID,
    E_FEATURE_CYCLE,
    E_SCHEMA_UNKNOWN_FIELD,
    E_TIMESTAMP_INVALID,
    load_event_file,
    validate_event,
    validate_feature_table,
    validate_label_definition,
)


def base_event(event_id: str = "e1", timestamp: int = 1) -> dict:
    return {
        "event_id": event_id,
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "view",
        "timestamp": timestamp,
        "context": {"price": 10.0},
    }


class ValidationTests(unittest.TestCase):
    def test_required_business_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            rows = [
                "not json",
                {**base_event("e1", 1), "unexpected": 1},
                base_event("e2", 2),
                base_event("e3", 2),
                base_event("e4", -1),
            ]
            path.write_text(
                "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            _, failures = load_event_file(path, {"view", "click", "purchase"})
        codes = {failure["code"] for failure in failures}
        self.assertIn(E_EVENT_INVALID, codes)
        self.assertIn(E_SCHEMA_UNKNOWN_FIELD, codes)
        self.assertIn(E_TIMESTAMP_INVALID, codes)

    def test_duplicate_unknown_type_and_non_finite(self) -> None:
        allowed = {"view"}
        failures = []
        seen: set[str] = set()
        for event in (
            base_event("e1"),
            {**base_event("e1", 2), "event_type": "teleport"},
            {**base_event("e2", 3), "context": {"price": float("nan")}},
        ):
            _, current = validate_event(event, allowed, seen)
            failures.extend(current)
            if not current:
                seen.add(event["event_id"])
        self.assertTrue(any(item["error_type"] == "DuplicateEventError" for item in failures))
        self.assertTrue(any(item["error_type"] == "EnumError" for item in failures))
        self.assertTrue(any(item["error_type"] == "NonFiniteNumberError" for item in failures))

    def test_cycle_detection(self) -> None:
        raw = {
            "features": [
                {
                    "name": "a",
                    "entity": "user",
                    "window_ms": 10,
                    "aggregation": "count",
                    "upstream_features": ["b"],
                },
                {
                    "name": "b",
                    "entity": "user",
                    "window_ms": 10,
                    "aggregation": "count",
                    "upstream_features": ["a"],
                },
            ]
        }
        table, failures = validate_feature_table(raw)
        self.assertIsNone(table)
        self.assertTrue(any(item["code"] == E_FEATURE_CYCLE for item in failures))

    def test_unknown_upstream_and_label_reference(self) -> None:
        raw = {
            "features": [
                {
                    "name": "a",
                    "entity": "user",
                    "window_ms": 10,
                    "aggregation": "count",
                    "upstream_features": ["missing"],
                }
            ]
        }
        table, failures = validate_feature_table(raw)
        self.assertIsNone(table)
        self.assertTrue(any(item["code"] == E_SCHEMA_UNKNOWN_FIELD for item in failures))

        valid_raw = {
            "features": [
                {
                    "name": "a",
                    "entity": "user",
                    "window_ms": 10,
                    "aggregation": "count",
                    "upstream_features": [],
                }
            ]
        }
        table, _ = validate_feature_table(valid_raw)
        self.assertIsNotNone(table)
        labels = {
            "name": "label",
            "trigger_event_types": ["view"],
            "delay_ms": 1,
            "feature_names": ["ghost"],
            "positive_rule": {"feature": "ghost", "operator": "ge", "value": 1},
            "negative_rule": {"else": True},
        }
        _, label_failures = validate_label_definition(labels, table)
        self.assertTrue(any(item["code"] == E_SCHEMA_UNKNOWN_FIELD for item in label_failures))


if __name__ == "__main__":
    unittest.main()

