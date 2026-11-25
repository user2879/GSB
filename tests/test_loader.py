import json
import unittest
from pathlib import Path

from ffbudget import errors
from ffbudget.loader import load_batch_file, load_config, load_requests

CONFIG = {
    "config_version": "t", "format_version": 1,
    "fixed_point": {"width_tier": "i64q16", "frac_bits": 16},
    "staleness": {
        "max_staleness_ms": 1000, "penalty_fresh_q16": 65536,
        "stale_penalty_per_ms_q16": 8, "stale_floor_q16": 16384,
        "penalty_unavailable_q16": 6554},
    "budget": {"score_point_budget": 100, "slot_point_quota": 50},
    "scheduler": {"max_concurrency": 2},
    "slots": [
        {"key": "r", "layer": "request", "dim": 2, "retention": 2,
         "weight": 3, "default_vector": [0, 0],
         "default_vector_source": "test://r"},
        {"key": "i", "layer": "item", "dim": 3, "retention": 1,
         "weight": 2, "default_vector": [1, 1, 1],
         "default_vector_source": "test://i"}],
}


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.id().replace(".", "_"))
        self.tmp.mkdir(exist_ok=True)
        self.config_path = self.tmp / "config.json"
        self.config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    def tearDown(self):
        for child in self.tmp.iterdir():
            child.unlink()
        self.tmp.rmdir()

    def test_config_validation_duplicate_and_bad_layer(self):
        bad = json.loads(json.dumps(CONFIG))
        bad["slots"].append(dict(bad["slots"][0]))
        bad["slots"][1]["layer"] = "unknown"
        path = self.tmp / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        _, failures = load_config(path)
        codes = {item["code"] for item in failures}
        self.assertIn(errors.E_DUPLICATE_SLOT_KEY, codes)
        self.assertIn(errors.E_UNKNOWN_LAYER, codes)

    def test_config_bad_retention_dimension(self):
        bad = json.loads(json.dumps(CONFIG))
        bad["slots"][0]["retention"] = 0
        bad["slots"][1]["dim"] = -1
        path = self.tmp / "bad2.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        _, failures = load_config(path)
        codes = {item["code"] for item in failures}
        self.assertIn(errors.E_BAD_RETENTION, codes)
        self.assertIn(errors.E_BAD_DIMENSION, codes)

    def test_batch_validation_rules(self):
        config, failures = load_config(self.config_path)
        self.assertEqual(failures, [])
        lines = [
            {"batch_seq": 1, "rev_seq": 1, "slot_key": "r",
             "effective_ms": 10, "points": 2, "vector": [1, 2]},
            {"batch_seq": 1, "rev_seq": 0, "slot_key": "r",
             "effective_ms": 11, "points": 2, "vector": [2, 2]},
            {"batch_seq": 1, "rev_seq": 2, "slot_key": "r",
             "effective_ms": -1, "points": 0, "vector": [1, 2]},
            {"batch_seq": 1, "rev_seq": 3, "slot_key": "r",
             "effective_ms": 12, "points": 0, "vector": [1, 2]},
            {"batch_seq": 1, "rev_seq": 4, "slot_key": "missing",
             "effective_ms": 5, "points": 1, "vector": [1]},
            {"batch_seq": 1, "rev_seq": 1, "slot_key": "i",
             "effective_ms": 5, "points": 3, "vector": [1, 2]},
        ]
        path = self.tmp / "b.ndjson"
        path.write_text("\n".join(json.dumps(x) for x in lines),
                        encoding="utf-8")
        slots = {slot.key: slot for slot in
                 load_config(self.config_path)[0] and _slots(config)}
        _, receipts, batch_failures = load_batch_file(path, slots)
        codes = [item["code"] for item in batch_failures]
        self.assertIn(errors.E_REV_SEQ_NOT_MONOTONIC, codes)
        self.assertIn(errors.E_BAD_EFFECTIVE_MS, codes)
        self.assertIn(errors.E_BAD_POINTS, codes)
        self.assertIn(errors.E_UNKNOWN_SLOT, codes)
        self.assertEqual(len(receipts), 1)

    def test_batch_seq_must_be_one_per_file(self):
        config, _ = load_config(self.config_path)
        slots = {slot.key: slot for slot in _slots(config)}
        lines = [
            {"batch_seq": 1, "rev_seq": 1, "slot_key": "r",
             "effective_ms": 10, "points": 2, "vector": [1, 2]},
            {"batch_seq": 2, "rev_seq": 2, "slot_key": "r",
             "effective_ms": 12, "points": 2, "vector": [3, 4]},
        ]
        path = self.tmp / "mixed.ndjson"
        path.write_text("\n".join(json.dumps(x) for x in lines),
                        encoding="utf-8")
        _, _, failures = load_batch_file(path, slots)
        self.assertEqual(
            [item["code"] for item in failures],
            [errors.E_BATCH_SEQ_NOT_STRICT])

    def test_request_validation(self):
        payload = {"requests": [
            {"request_id": "q", "rank_ms": 9, "candidates": ["a", "b"]},
            {"request_id": "q", "rank_ms": 9, "candidates": ["a"]},
            {"request_id": "bad", "rank_ms": -1, "candidates": ["a"]},
            {"request_id": "dup", "rank_ms": 9, "candidates": ["a", "a"]},
        ]}
        path = self.tmp / "requests.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        requests, failures = load_requests(path)
        self.assertEqual(len(requests), 1)
        self.assertTrue(failures)


def _slots(config):
    from ffbudget.loader import _slots_from_config
    return _slots_from_config(config).values()


if __name__ == "__main__":
    unittest.main()
