from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quota_app.codec import parse_document
from quota_app.derive import resolve_effective
from quota_app.engine import JudgmentEngine, replay_events
from quota_app.errors import E_FORMAT, E_ORDER, ConfigFormatError, OrderError
from quota_app.events import load_events
from quota_app.models import (
    AccessEvent,
    DECISION_ALLOW,
    DECISION_OVER_QUOTA,
    DECISION_RATE_LIMITED,
)

from .helpers import document, event_records, write_events


class EngineTests(unittest.TestCase):
    def test_sample_decisions_and_error_bounds(self) -> None:
        effective = resolve_effective(document())
        judgments, usage = replay_events(effective, [AccessEvent(**record) for record in event_records()])
        self.assertEqual(
            [judgment.decision for judgment in judgments],
            [
                DECISION_ALLOW,
                DECISION_ALLOW,
                DECISION_RATE_LIMITED,
                DECISION_ALLOW,
                DECISION_OVER_QUOTA,
                DECISION_ALLOW,
                DECISION_ALLOW,
                DECISION_OVER_QUOTA,
                DECISION_ALLOW,
                DECISION_RATE_LIMITED,
            ],
        )
        self.assertTrue(all(judgment.single_error_upper == 0 for judgment in judgments))
        self.assertTrue(all(judgment.aggregate_error_upper == 0 for judgment in judgments))
        self.assertEqual(usage["root"].cumulative_write_bytes, 400)
        self.assertEqual(usage["root"].rate_limited_events, 1)
        self.assertEqual(usage["child"].over_quota_events, 1)

    def test_capacity_boundary_is_inclusive(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {"id": "a", "capacity_bytes": 100, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes"}
                ],
            }
        )
        engine = JudgmentEngine(resolve_effective(parsed))
        first = engine.evaluate(AccessEvent(1, 0, "a", 1, 100))
        second = engine.evaluate(AccessEvent(2, 0, "a", 1, 1))
        self.assertEqual(first.decision, DECISION_ALLOW)
        self.assertEqual(second.decision, DECISION_OVER_QUOTA)

    def test_token_bucket_refill_exactness(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {"id": "a", "capacity_bytes": 1000, "ops_per_sec": 1, "burst_capacity": 1, "limit_dimension": "operations"}
                ],
            }
        )
        engine = JudgmentEngine(resolve_effective(parsed))
        self.assertEqual(engine.evaluate(AccessEvent(1, 0, "a", 1, 0)).decision, DECISION_ALLOW)
        limited = engine.evaluate(AccessEvent(2, 999, "a", 1, 0))
        self.assertEqual(limited.decision, DECISION_RATE_LIMITED)
        self.assertEqual(limited.tokens_scaled, 999)
        self.assertEqual(engine.evaluate(AccessEvent(3, 1000, "a", 1, 0)).decision, DECISION_ALLOW)

    def test_v1_zero_rate_and_burst_disables_rate_limit(self) -> None:
        parsed = parse_document(
            {
                "format_version": 1,
                "volumes": [{"id": "a", "capacity_bytes": 1000}],
            }
        )
        engine = JudgmentEngine(resolve_effective(parsed))
        for seq in range(1, 20):
            self.assertEqual(engine.evaluate(AccessEvent(seq, seq, "a", 1, 1)).decision, DECISION_ALLOW)

    def test_resume_from_snapshot_matches_full_replay(self) -> None:
        effective = resolve_effective(document())
        all_events = [AccessEvent(**record) for record in event_records()]
        full, full_usage = replay_events(effective, all_events)
        first, snapshot = replay_events(effective, all_events[:4])
        second, resumed_usage = replay_events(effective, all_events[4:], snapshot)
        combined = first + second
        self.assertEqual([row.to_json() for row in combined], [row.to_json() for row in full])
        self.assertEqual(
            {key: value.to_json() for key, value in resumed_usage.items()},
            {key: value.to_json() for key, value in full_usage.items()},
        )

    def test_event_timestamp_regression_is_e_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.ndjson"
            records = event_records()
            records[4]["timestamp_ms"] = 50
            write_events(path, records)
            with self.assertRaises(OrderError) as context:
                load_events(path)
            self.assertEqual(context.exception.code, E_ORDER)
            self.assertEqual(context.exception.details["seq"], 5)

    def test_malformed_event_is_e_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.ndjson"
            path.write_text('{"seq": 1, "timestamp_ms": 0}\n', encoding="utf-8")
            with self.assertRaises(ConfigFormatError) as context:
                load_events(path)
            self.assertEqual(context.exception.code, E_FORMAT)


if __name__ == "__main__":
    unittest.main()
