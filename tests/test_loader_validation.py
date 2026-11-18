import json
import os
import tempfile
import unittest

from ab_workbench.loader import load_exposures
from ab_workbench.errors import E_LOG_INVALID


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def good_row(**over):
    row = {
        "request_id": "r1",
        "group": "c",
        "user_id": "u1",
        "item_id": "i1",
        "position": 1,
        "score": 0.5,
        "clicked": 0,
        "converted": 1,
        "event_ts_ms": 10,
    }
    row.update(over)
    return row


class LoaderValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "log.jsonl")

    def _load(self, rows):
        write_jsonl(self.path, rows)
        return load_exposures(self.path, {"c", "t"})

    def test_valid_row(self):
        result = self._load([good_row()])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.exposures), 1)

    def test_non_finite_score(self):
        result = self._load([good_row(score=float("nan"))])
        self.assertEqual(result.failures[0].code, E_LOG_INVALID)
        self.assertEqual(result.failures[0].reason, "score_not_finite")

    def test_group_out_of_range(self):
        result = self._load([good_row(group="zzz")])
        self.assertEqual(result.failures[0].reason, "group_out_of_range")

    def test_bad_click_flag(self):
        result = self._load([good_row(clicked=2)])
        self.assertEqual(result.failures[0].reason, "clicked_not_binary")

    def test_position_must_be_positive(self):
        result = self._load([good_row(position=0)])
        self.assertEqual(result.failures[0].reason, "position_not_positive_int")

    def test_duplicate_request_item_rejected(self):
        rows = [good_row(), good_row(event_ts_ms=11)]
        result = self._load(rows)
        self.assertTrue(any(f.reason == "duplicate_request_item" for f in result.failures))

    def test_monotonic_timestamps(self):
        rows = [
            good_row(item_id="i1", event_ts_ms=100),
            good_row(item_id="i2", event_ts_ms=50),
        ]
        result = self._load(rows)
        self.assertTrue(
            any(f.reason == "timestamp_not_monotonic_within_request" for f in result.failures)
        )

    def test_json_parse_error(self):
        with open(self.path, "w") as handle:
            handle.write("{not valid json}\n")
        result = load_exposures(self.path, {"c", "t"})
        self.assertEqual(result.failures[0].reason, "json_parse_error")


if __name__ == "__main__":
    unittest.main()
