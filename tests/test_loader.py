import json
import tempfile
import unittest
from pathlib import Path

from graymirror.engine.errors import E_RULE, E_STREAM, GrayError
from graymirror.engine.loader import load_rules, load_stream, match_rule

VALID_RULE = {
    "rule_id": "r1", "prefix": "/api", "stable_target": "s",
    "gray_target": "g", "gray_weight": 1000, "mirror_rate": 2000,
    "gray_retries": 1, "mirror_timeout_ms": 100,
}


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _rules(self, *rules):
        path = self.root / "rules.json"
        path.write_text(json.dumps({"rules": list(rules)}))
        return path

    def _stream(self, rows):
        path = self.root / "stream.ndjson"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows))
        return path

    def test_valid_load(self):
        rules = load_rules(self._rules(VALID_RULE))
        self.assertEqual(rules[0].rule_id, "r1")
        rows = self._stream([
            {"request_id": "q1", "path": "/api/x", "client_key": "c",
             "timestamp_ms": 1},
            {"request_id": "q2", "path": "/api/y", "client_key": "c",
             "timestamp_ms": 2},
        ])
        stream = load_stream(rows, rules)
        self.assertEqual(len(stream), 2)

    def test_duplicate_rule_id_rejected(self):
        with self.assertRaises(GrayError) as ctx:
            load_rules(self._rules(VALID_RULE, dict(VALID_RULE)))
        self.assertEqual(ctx.exception.code, E_RULE)
        self.assertIn("rule_id", ctx.exception.field_path)

    def test_prefix_must_start_with_slash(self):
        bad = dict(VALID_RULE, rule_id="r2", prefix="api")
        with self.assertRaises(GrayError) as ctx:
            load_rules(self._rules(bad))
        self.assertEqual(ctx.exception.code, E_RULE)

    def test_weight_range_rejected_with_line(self):
        bad = dict(VALID_RULE, gray_weight=10001)
        with self.assertRaises(GrayError) as ctx:
            load_rules(self._rules(bad))
        self.assertEqual(ctx.exception.code, E_RULE)
        self.assertIsNotNone(ctx.exception.line)

    def test_negative_retry_rejected(self):
        bad = dict(VALID_RULE, gray_retries=-1)
        with self.assertRaises(GrayError) as ctx:
            load_rules(self._rules(bad))
        self.assertEqual(ctx.exception.code, E_RULE)

    def test_timestamp_regression_rejected(self):
        rules = load_rules(self._rules(VALID_RULE))
        rows = self._stream([
            {"request_id": "q1", "path": "/api/x", "client_key": "c",
             "timestamp_ms": 10},
            {"request_id": "q2", "path": "/api/x", "client_key": "c",
             "timestamp_ms": 9},
        ])
        with self.assertRaises(GrayError) as ctx:
            load_stream(rows, rules)
        self.assertEqual(ctx.exception.code, E_STREAM)
        self.assertEqual(ctx.exception.line, 2)

    def test_unmatched_path_rejected(self):
        rules = load_rules(self._rules(VALID_RULE))
        rows = self._stream([
            {"request_id": "q1", "path": "/web/x", "client_key": "c",
             "timestamp_ms": 1},
        ])
        with self.assertRaises(GrayError) as ctx:
            load_stream(rows, rules)
        self.assertEqual(ctx.exception.code, E_STREAM)

    def test_duplicate_request_rejected(self):
        rules = load_rules(self._rules(VALID_RULE))
        rows = self._stream([
            {"request_id": "q", "path": "/api/x", "client_key": "c",
             "timestamp_ms": 1},
            {"request_id": "q", "path": "/api/x", "client_key": "c",
             "timestamp_ms": 2},
        ])
        with self.assertRaises(GrayError) as ctx:
            load_stream(rows, rules)
        self.assertEqual(ctx.exception.code, E_STREAM)

    def test_longest_prefix_wins(self):
        rules = load_rules(self._rules(
            VALID_RULE,
            dict(VALID_RULE, rule_id="r2", prefix="/api/orders"),
        ))
        self.assertEqual(match_rule(rules, "/api/orders/1").rule_id, "r2")
        self.assertEqual(match_rule(rules, "/api/users").rule_id, "r1")


if __name__ == "__main__":
    unittest.main()
