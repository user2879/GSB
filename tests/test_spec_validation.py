import json
import os
import tempfile
import unittest

from delayqueue import SpecError, load_declaration, load_stream
from delayqueue.models import payload_digest


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "topics.json")
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump({
                "spec_version": "v1",
                "global_concurrency": 2,
                "topics": {
                    "a": {"visibility_timeout_ms": 100,
                          "max_deliveries": 3,
                          "backoff_ms": [10, 20, 30],
                          "concurrency": 1},
                },
            }, handle)
        self.spec = load_declaration(self.spec_path)

    def _write_stream(self, lines):
        path = os.path.join(self.tmp, "messages.ndjson")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path

    def test_valid_stream_loads(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"a","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}',
        ])
        messages = load_stream(path, self.spec)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].payload, b"\x01")

    def test_unknown_topic_rejected_with_line_and_field(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"ghost","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}',
        ])
        with self.assertRaises(SpecError) as caught:
            load_stream(path, self.spec)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.line, 1)
        self.assertEqual(caught.exception.field, "topic")

    def test_negative_deliver_at_rejected(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"a","deliver_at":-5,'
            '"payload":1,"idempotency_key":"k1"}',
        ])
        with self.assertRaises(SpecError) as caught:
            load_stream(path, self.spec)
        self.assertEqual(caught.exception.field, "deliver_at")

    def test_same_idempotency_different_payload_rejected(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"a","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}',
            '{"message_id":"m2","topic":"a","deliver_at":0,'
            '"payload":2,"idempotency_key":"k1"}',
        ])
        with self.assertRaises(SpecError) as caught:
            load_stream(path, self.spec)
        self.assertEqual(caught.exception.line, 2)
        self.assertEqual(caught.exception.field, "idempotency_key")

    def test_same_idempotency_same_payload_allowed(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"a","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}',
            '{"message_id":"m2","topic":"a","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}',
        ])
        messages = load_stream(path, self.spec)
        self.assertEqual(len(messages), 2)

    def test_backoff_shorter_than_max_deliveries_minus_one_rejected(self):
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump({
                "global_concurrency": 2,
                "topics": {"a": {"visibility_timeout_ms": 100,
                                  "max_deliveries": 4,
                                  "backoff_ms": [10],
                                  "concurrency": 1}},
            }, handle)
        with self.assertRaises(SpecError) as caught:
            load_declaration(self.spec_path)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("backoff_ms", caught.exception.field)

    def test_duplicate_topic_id_rejected(self):
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump({
                "global_concurrency": 2,
                "topics": [
                    {"topic": "a", "visibility_timeout_ms": 100,
                     "max_deliveries": 2, "backoff_ms": [10],
                     "concurrency": 1},
                    {"topic": "a", "visibility_timeout_ms": 100,
                     "max_deliveries": 2, "backoff_ms": [10],
                     "concurrency": 1},
                ],
            }, handle)
        with self.assertRaises(SpecError):
            load_declaration(self.spec_path)

    def test_boolean_is_not_an_integer(self):
        path = self._write_stream([
            '{"message_id":"m1","topic":"a","deliver_at":true,'
            '"payload":1,"idempotency_key":"k1"}',
        ])
        with self.assertRaises(SpecError) as caught:
            load_stream(path, self.spec)
        self.assertEqual(caught.exception.field, "deliver_at")


if __name__ == "__main__":
    unittest.main()
