from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from driftbench.errors import ConfigAbort
from driftbench.loading import load_workspace
from tests.common import copy_samples


class LoadingTests(unittest.TestCase):
    def test_invalid_exposure_rows_are_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = copy_samples(Path(temporary) / "sample")
            valid = {
                "request_id": "req-000001",
                "experiment_group": "control",
                "user_id": "u0001",
                "item_id": "item-001",
                "rank": 1,
                "model_score": 0.1,
                "clicked": 0,
                "converted": 0,
                "event_ms": 1,
            }
            bad = [
                {**valid, "request_id": "bad-nonfinite", "model_score": float("nan")},
                {**valid, "request_id": "bad-group", "experiment_group": "other"},
                valid,
            ]
            from tests.common import append_jsonl

            append_jsonl(root / "exposures.jsonl", bad)
            state = load_workspace(root)
            log_failures = [f for f in state.failures if f.code == "E_LOG_INVALID"]
            self.assertEqual(len(state.exposures), 480)
            self.assertEqual(len(log_failures), 3)
            self.assertEqual(
                {failure.request_id for failure in log_failures},
                {"bad-nonfinite", "bad-group", "req-000001"},
            )
            self.assertEqual(
                {failure.field_path for failure in log_failures},
                {"model_score", "experiment_group", "request_id"},
            )

    def test_invalid_spec_rows_are_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = copy_samples(Path(temporary) / "sample")
            invalid_spec = {
                "spec_version": "v-bad",
                "metric_name": "ctr",
                "truncation_rank": 5,
                "normalization": "none",
                "dedup_count": False,
                "sample_filter": [{"field": "not_a_field", "op": "eq", "value": 1}],
            }
            with (root / "specs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(invalid_spec) + "\n")
            state = load_workspace(root)
            self.assertEqual(sorted(state.specs), ["v1", "v2", "v3"])
            failures = [f for f in state.failures if f.code == "E_SPEC_INVALID"]
            self.assertTrue(any("unknown exposure field" in f.message for f in failures))
            self.assertTrue(any(f.field_path == "sample_filter[0].field" for f in failures))

    def test_config_validation_aborts_with_field_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = copy_samples(Path(temporary) / "sample")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["bucket_count"] = 0
            config["active_spec_version"] = "missing"
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigAbort) as caught:
                load_workspace(root)
            paths = {failure.field_path for failure in caught.exception.failures}
            self.assertIn("bucket_count", paths)
            self.assertIn("active_spec_version", paths)
            self.assertTrue(all(f.code == "E_CONFIG_INVALID" for f in caught.exception.failures))

    def test_config_rejects_zero_tolerance_and_bad_attribution_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = copy_samples(Path(temporary) / "sample")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["drift_tolerance"] = "0"
            config["attribution_order"] = ["truncation_rank", "bogus"]
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigAbort) as caught:
                load_workspace(root)
            paths = {failure.field_path for failure in caught.exception.failures}
            self.assertEqual(paths, {"drift_tolerance", "attribution_order"})


if __name__ == "__main__":
    unittest.main()
