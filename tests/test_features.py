from __future__ import annotations

import unittest

from feature_pipeline.compute import compute_feature_records
from feature_pipeline.validation import validate_feature_table


def event(event_id: str, timestamp: int, event_type: str = "view", item: str = "i1") -> dict:
    return {
        "event_id": event_id,
        "user_id": "u1",
        "item_id": item,
        "event_type": event_type,
        "timestamp": timestamp,
        "context": {"price": float(timestamp), "dwell_ms": timestamp + 1},
    }


class FeatureTests(unittest.TestCase):
    def test_inclusive_window_empty_marker_and_cache_key(self) -> None:
        raw = {
            "features": [
                {
                    "name": "price_sum",
                    "entity": "user",
                    "window_ms": 10000,
                    "aggregation": "sum",
                    "source_field": "context.price",
                    "empty_value": -1,
                    "upstream_features": [],
                },
                {
                    "name": "purchase_count",
                    "entity": "user",
                    "window_ms": 5000,
                    "aggregation": "count",
                    "empty_value": 0,
                    "filter": {"field": "event_type", "equals": "purchase"},
                    "upstream_features": [],
                },
            ]
        }
        table, failures = validate_feature_table(raw)
        self.assertEqual(failures, [])
        assert table is not None
        events = [event("e1", 1000), event("e2", 11000)]
        records, failures = compute_feature_records(table.definitions["price_sum"], events, {})
        self.assertEqual(failures, [])
        by_event = {record["event_id"]: record for record in records}
        self.assertEqual(by_event["e2"]["value"], 12000.0)
        self.assertFalse(by_event["e2"]["empty"])
        repeated, _ = compute_feature_records(table.definitions["price_sum"], events, {})
        self.assertEqual([item["cache_key"] for item in records], [item["cache_key"] for item in repeated])

        purchase_records, _ = compute_feature_records(
            table.definitions["purchase_count"], events, {}
        )
        self.assertTrue(all(record["empty"] for record in purchase_records))
        self.assertTrue(all(record["value"] == 0 for record in purchase_records))

    def test_fixed_latest_order_for_same_timestamp(self) -> None:
        raw = {
            "features": [
                {
                    "name": "latest_price",
                    "entity": "user",
                    "window_ms": 1,
                    "aggregation": "latest",
                    "source_field": "context.price",
                    "empty_value": -1,
                    "upstream_features": [],
                }
            ]
        }
        table, _ = validate_feature_table(raw)
        assert table is not None
        first = event("e1", 1000)
        second = event("e2", 1000)
        first["context"]["price"] = 11.0
        second["context"]["price"] = 22.0
        records, _ = compute_feature_records(
            table.definitions["latest_price"], [second, first], {}
        )
        by_event = {record["event_id"]: record for record in records}
        self.assertEqual(by_event["e2"]["value"], 22.0)


if __name__ == "__main__":
    unittest.main()

