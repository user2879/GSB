from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eventbus_diag.batching import build_plan
from eventbus_diag.errors import SpecError
from eventbus_diag.spec import load_dataset

from .helpers import default_channels, default_subscribers, write_events, write_spec


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_dataset_and_deterministic_plan(self) -> None:
        spec = write_spec(
            self.root / "spec.json",
            subscribers=default_subscribers(2),
            channels=default_channels(batch_size=2, quota=2),
        )
        events = write_events(self.root / "events.ndjson", 5)
        dataset = load_dataset(spec, events)
        plan1 = build_plan(dataset.declaration, dataset.events)
        plan2 = build_plan(dataset.declaration, tuple(reversed(dataset.events)))
        self.assertEqual(plan1.digest(), plan2.digest())
        self.assertEqual(
            [batch.batch_id for batch in plan1.batches],
            [batch.batch_id for batch in plan2.batches],
        )
        self.assertEqual(
            [batch.canonical_content() for batch in plan1.batches],
            [batch.canonical_content() for batch in plan2.batches],
        )
        for subscriber in dataset.declaration.subscribers:
            ids = sorted(
                event_id
                for batch in plan1.batches
                if batch.subscriber_id == subscriber.subscriber_id
                for event_id in batch.event_ids
            )
            self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_duplicate_port_reports_first_field_and_line(self) -> None:
        subscribers = default_subscribers(2)
        subscribers[1]["port"] = subscribers[0]["port"]
        spec = write_spec(self.root / "spec.json", subscribers=subscribers)
        events = write_events(self.root / "events.ndjson", 1)
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, events)
        error = raised.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.field, "subscribers[1].port")
        self.assertEqual(error.line, 17)
        text = spec.read_text(encoding="utf-8").splitlines()[error.line - 1]
        self.assertIn('"port"', text)

    def test_non_loopback_host_and_nonpositive_budget(self) -> None:
        subscribers = default_subscribers(1)
        subscribers[0]["host"] = "localhost"
        spec = write_spec(self.root / "spec.json", subscribers=subscribers)
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, write_events(self.root / "events.ndjson", 1))
        self.assertEqual(raised.exception.field, "subscribers[0].host")
        self.assertIn("127.0.0.1", raised.exception.message)

        subscribers = default_subscribers(1)
        subscribers[0]["processing_budget_ms"] = 0
        spec = write_spec(self.root / "spec2.json", subscribers=subscribers)
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, write_events(self.root / "events2.ndjson", 1))
        self.assertEqual(raised.exception.field, "subscribers[0].processing_budget_ms")

    def test_event_gap_and_timestamp_are_rejected_on_first_line(self) -> None:
        spec = write_spec(self.root / "spec.json")
        events = self.root / "events.ndjson"
        events.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "seq": 1,
                            "ts_ms": 100,
                            "topic": "orders",
                            "payload_digest": "x",
                            "expected_order": 1,
                        }
                    ),
                    json.dumps(
                        {
                            "seq": 3,
                            "ts_ms": 101,
                            "topic": "orders",
                            "payload_digest": "x",
                            "expected_order": 2,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, events)
        self.assertEqual(raised.exception.field, "seq")
        self.assertEqual(raised.exception.line, 2)

        events.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "seq": 1,
                            "ts_ms": 200,
                            "topic": "orders",
                            "payload_digest": "x",
                            "expected_order": 1,
                        }
                    ),
                    json.dumps(
                        {
                            "seq": 2,
                            "ts_ms": 199,
                            "topic": "orders",
                            "payload_digest": "x",
                            "expected_order": 2,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, events)
        self.assertEqual(raised.exception.field, "ts_ms")
        self.assertEqual(raised.exception.line, 2)

    def test_unknown_event_topic_is_rejected(self) -> None:
        spec = write_spec(self.root / "spec.json")
        events = self.root / "events.ndjson"
        events.write_text(
            json.dumps(
                {
                    "seq": 1,
                    "ts_ms": 100,
                    "topic": "refunds",
                    "payload_digest": "x",
                    "expected_order": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, events)
        self.assertEqual(raised.exception.field, "topic")
        self.assertEqual(raised.exception.line, 1)

    def test_invalid_json_reports_json_line(self) -> None:
        spec = write_spec(self.root / "spec.json")
        events = self.root / "events.ndjson"
        events.write_text('{"seq":1,\n', encoding="utf-8")
        with self.assertRaises(SpecError) as raised:
            load_dataset(spec, events)
        self.assertEqual(raised.exception.field, "$event_json")
        self.assertEqual(raised.exception.line, 1)


if __name__ == "__main__":
    unittest.main()
