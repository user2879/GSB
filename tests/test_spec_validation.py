import json
import tempfile
import unittest
from pathlib import Path

from elastic_scaler.errors import BudgetError, SpecError
from elastic_scaler.spec import load_spec, parse_spec

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cluster_v1.json"


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _load(self, raw):
        path = Path(self.tmp.name) / "spec.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_spec(path)

    def test_valid_spec_loads_with_stable_service_order(self):
        spec = self._load(self.raw)
        self.assertEqual([s.name for s in spec.services], ["asr", "embedding", "llm"])
        self.assertEqual(spec.horizon, 8)
        self.assertEqual(len(spec.digest), 64)

    def test_duplicate_service_name_rejected_with_field_paths(self):
        self.raw["services"][1]["name"] = "asr"
        with self.assertRaises(SpecError) as caught:
            self._load(self.raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertTrue(any("/name" in field for field in caught.exception.fields))

    def test_min_greater_than_max_rejected(self):
        self.raw["services"][0]["min_replicas"] = 5
        with self.assertRaises(SpecError) as caught:
            self._load(self.raw)
        self.assertIn("/services/0/min_replicas", caught.exception.fields)

    def test_negative_request_and_bool_rejected(self):
        self.raw["services"][0]["requests"][2] = -1
        with self.assertRaises(SpecError):
            self._load(self.raw)
        self.raw["services"][0]["requests"][2] = True
        with self.assertRaises(SpecError):
            self._load(self.raw)

    def test_non_positive_budget_is_budget_error(self):
        self.raw["global_replica_budget"] = 0
        with self.assertRaises(BudgetError) as caught:
            self._load(self.raw)
        self.assertEqual(caught.exception.code, "E_BUDGET")

    def test_infeasible_minimum_reserve_rejected(self):
        self.raw["global_replica_budget"] = 1
        with self.assertRaises(BudgetError) as caught:
            self._load(self.raw)
        self.assertIn("/global_replica_budget", caught.exception.fields)

    def test_quota_smaller_than_minimum_rejected(self):
        svc = self.raw["services"][0]
        svc["concurrency_quota"] = 1
        with self.assertRaises(BudgetError):
            self._load(self.raw)

    def test_bad_threshold_and_missing_field_rejected(self):
        self.raw["services"][0]["scale_threshold_pct"] = 120
        with self.assertRaises(SpecError):
            self._load(self.raw)
        del self.raw["services"][0]["request_ms"]
        with self.assertRaises(SpecError) as caught:
            self._load(self.raw)
        self.assertTrue(any("request_ms" in f for f in caught.exception.fields))

    def test_digest_deterministic_under_key_reordering(self):
        reordered = json.loads(json.dumps(self.raw))
        reordered["services"][0] = dict(reversed(list(reordered["services"][0].items())))
        self.assertEqual(parse_spec(self.raw).digest, parse_spec(reordered).digest)


if __name__ == "__main__":
    unittest.main()
