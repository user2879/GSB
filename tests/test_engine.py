from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixedprec_diag.config import parse_config
from mixedprec_diag.deterministic import generate_artifacts
from mixedprec_diag.engine import InferenceEngine, write_report
from mixedprec_diag.errors import E_OOM, E_PRESSURE
from tests.helpers import config_payload


class EngineSchedulingTests(unittest.TestCase):
    def test_capacity_split_queue_pressure_and_replay(self) -> None:
        artifacts = generate_artifacts(55, 400, 64, 256)
        config = parse_config(
            config_payload(
                memory_budget_mib=1,
                block_size_kb=4,
                precision="float32",
                batch_size_limit=400,
                preallocation="per_batch",
                pressure_threshold_ms=0.0,
            )
        )
        report = InferenceEngine().run(config, artifacts, config_version=1)
        self.assertEqual(len(report["samples"]), 400)
        self.assertEqual(len(report["batches"]), 2)
        self.assertEqual([batch["sample_count"] for batch in report["batches"]], [307, 93])
        self.assertTrue(any(event["queue_reason"] == "CAPACITY_SPLIT" for event in report["queue_events"]))
        self.assertIn(E_PRESSURE, report["summary"]["codes"])
        self.assertTrue(report["memory"]["replay"]["verified"], report["memory"]["replay"]["errors"])
        self.assertLessEqual(report["memory"]["peak_used_blocks"], config.total_blocks)
        self.assertGreaterEqual(len(report["pressure"]["intervals"]), 1)
        for event in report["memory"]["events"]:
            self.assertLessEqual(len(event["used_block_ids_after"]), config.total_blocks)

    def test_model_weight_oom_is_reported_without_inference(self) -> None:
        artifacts = generate_artifacts(5, 4, 8, 512)
        config = parse_config(
            config_payload(
                memory_budget_mib=1,
                block_size_kb=64,
                precision="fixed",
                batch_size_limit=2,
                preallocation="per_batch",
            )
        )
        report = InferenceEngine().run(config, artifacts, config_version=1)
        self.assertIn(E_OOM, report["summary"]["codes"])
        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertEqual(report["summary"]["samples_completed"], 0)
        self.assertIsNotNone(report["summary"]["fatal_error"])
        self.assertEqual(report["memory"]["events"][0]["action"], "oom")
        self.assertTrue(report["memory"]["replay"]["verified"])

    def test_report_round_trips_without_nonfinite_json(self) -> None:
        artifacts = generate_artifacts(99, 7, 6, 10)
        config = parse_config(config_payload(batch_size_limit=3))
        report = InferenceEngine().run(config, artifacts, config_version=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_report(path, report)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["run_id"], report["run_id"])
        self.assertEqual(loaded["summary"]["status"], report["summary"]["status"])
        self.assertIn("curve", loaded["memory"])
        self.assertIn("queries", loaded["ui_latency"])


if __name__ == "__main__":
    unittest.main()
