import json
import unittest

from alert_pipeline.codec import (FILL_RULES, decode_record, encode_record)
from alert_pipeline.errors import FieldError

SERVICES = {"svc-a", "svc-b"}


class CodecTest(unittest.TestCase):
    def test_v1_normalization_is_fully_filled_and_traceable(self):
        raw = json.dumps({
            "version": "v1", "timestamp_ms": 100, "service_id": "svc-a",
            "severity": "major"})
        alert = decode_record(raw, 1, 1000, SERVICES)
        self.assertEqual(alert.source_version, "v1")
        self.assertEqual(
            set(alert.filled),
            {"fingerprint", "silence_window_ms", "labels",
             "root_cause_refs", "strategy_id"})
        for field_name in alert.filled:
            self.assertIn(("v1", field_name), FILL_RULES)
        self.assertEqual(alert.silence_window_ms, 1000)
        self.assertEqual(alert.strategy_id, "strategy_default")
        # Round-trip to the originating wire version.
        wire = encode_record(alert)
        self.assertEqual(set(wire),
                         {"version", "timestamp_ms", "service_id",
                          "severity"})

    def test_v2_v3_fill_rules(self):
        v2 = decode_record(json.dumps({
            "version": "v2", "timestamp_ms": 1, "service_id": "svc-a",
            "severity": "info", "fingerprint": "fp_0123456789ab",
            "silence_window_ms": 0}), 2, 1000, SERVICES)
        self.assertEqual(set(v2.filled),
                         {"labels", "root_cause_refs", "strategy_id"})
        v3 = decode_record(json.dumps({
            "version": "v3", "timestamp_ms": 2, "service_id": "svc-b",
            "severity": "info", "fingerprint": "fp_0123456789ab",
            "silence_window_ms": 5, "root_cause_refs": ["svc-a"],
            "strategy_id": "strategy_x"}), 3, 1000, SERVICES)
        self.assertEqual(v3.filled, ("labels",))
        self.assertEqual(v3.root_cause_refs, ("svc-a",))

    def test_order_independent_byte_identical_normalization(self):
        raw = json.dumps({
            "version": "v3", "timestamp_ms": 7, "service_id": "svc-a",
            "severity": "minor", "fingerprint": "fp_0123456789ab",
            "silence_window_ms": 9, "root_cause_refs": ["svc-a", "svc-b"],
            "strategy_id": "s", "labels": {"z": "1", "a": "2"}})
        first = decode_record(raw, 1, 1000, SERVICES)
        second = decode_record(raw, 99, 1000, SERVICES)
        # Logical content bytes identical; only provenance line differs.
        self.assertEqual(first.canonical(), second.canonical())
        self.assertNotIn(b'"line"', first.canonical())

    def test_rejected_cases_report_code_line_field(self):
        cases = [
            ('{"version":"v9","timestamp_ms":1,"service_id":"svc-a",'
             '"severity":"info"}', "version"),
            ('{"version":"v2","service_id":"svc-a","severity":"info",'
             '"fingerprint":"fp_x","silence_window_ms":1}', "timestamp_ms"),
            ('{"version":"v2","timestamp_ms":1,"service_id":"svc-a",'
             '"severity":"info","fingerprint":"bad","silence_window_ms":1}',
             "fingerprint"),
            ('{"version":"v2","timestamp_ms":1,"service_id":"svc-a",'
             '"severity":"info","fingerprint":"fp_01234567",'
             '"silence_window_ms":-2}', "silence_window_ms"),
            ('{"version":"v3","timestamp_ms":1,"service_id":"svc-a",'
             '"severity":"info","fingerprint":"fp_01234567",'
             '"silence_window_ms":1,"root_cause_refs":["svc-ghost"],'
             '"strategy_id":"s"}', "root_cause_refs"),
        ]
        for text, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(FieldError) as ctx:
                    decode_record(text, 5, 1000, SERVICES)
                self.assertEqual(ctx.exception.code, "E_FIELD")
                self.assertEqual(ctx.exception.line, 5)
                self.assertEqual(ctx.exception.field, field)


if __name__ == "__main__":
    unittest.main()
