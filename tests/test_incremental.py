from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_pipeline.compare import compare_states
from feature_pipeline.engine import PipelineEngine
from feature_pipeline.state import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def run(engine: PipelineEngine, events: str, features: str, labels: str, **kwargs):
    return engine.run(
        config_path=kwargs.pop("config", SAMPLES / "config.json"),
        events_path=SAMPLES / events,
        features_path=SAMPLES / features,
        labels_path=SAMPLES / labels,
        **kwargs,
    )


class IncrementalEquivalenceTests(unittest.TestCase):
    def test_edit_sequence_matches_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incremental = PipelineEngine(root / "incremental")
            _, code = run(
                incremental,
                "events_base_10.jsonl",
                "features_base.json",
                "labels_base.json",
                reset=True,
            )
            self.assertEqual(code, 0)
            run(incremental, "events_pre_backfill.jsonl", "features_base.json", "labels_base.json")
            window_report, _ = run(
                incremental,
                "events_pre_backfill.jsonl",
                "features_window_changed.json",
                "labels_base.json",
            )
            self.assertEqual(
                window_report["invalidation_ledger"]["invalidated_features"],
                ["item_view_count_5s"],
            )
            self.assertIn(
                "item_avg_price_10s", window_report["invalidation_ledger"]["reused_features"]
            )
            run(incremental, "events_pre_backfill.jsonl", "features.json", "labels.json")
            backfill_report, _ = run(
                incremental,
                "events.jsonl",
                "features.json",
                "labels.json",
                edits_path=SAMPLES / "edits.json",
            )
            self.assertIn(
                "item_view_count_5s",
                backfill_report["invalidation_ledger"]["reused_features"],
            )
            self.assertIn(
                "item_avg_price_10s",
                backfill_report["invalidation_ledger"]["invalidated_features"],
            )
            self.assertTrue(backfill_report["invalidation_ledger"]["regenerated_samples"])
            maintenance_report, _ = run(
                incremental,
                "events.jsonl",
                "features.json",
                "labels.json",
            )
            self.assertEqual(len(maintenance_report["backfill_ledger"]), 1)
            self.assertEqual(
                maintenance_report["backfill_ledger"][0]["event_id"], "e0010"
            )
            self.assertTrue(
                maintenance_report["backfill_ledger"][0]["affected_features"]
            )
            self.assertTrue(
                maintenance_report["backfill_ledger"][0]["affected_samples"]
            )

            full = PipelineEngine(root / "full")
            full_report, _ = run(
                full, "events.jsonl", "features.json", "labels.json", reset=True
            )
            incremental_state, _ = SnapshotStore(root / "incremental").recover()
            full_state, _ = SnapshotStore(root / "full").recover()
            comparison = compare_states(incremental_state, full_state)
            self.assertTrue(comparison["equal"], comparison)
            self.assertTrue(comparison["feature_layer_equal"])
            self.assertTrue(comparison["sample_layer_equal"])
            self.assertTrue(comparison["cache_ledger_equal"])
            self.assertEqual(
                backfill_report["feature_snapshot_version"],
                full_report["feature_snapshot_version"],
            )
            self.assertGreater(full_report["empty_window_feature_count"], 0)

    def test_parallel_and_serial_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parallel = PipelineEngine(root / "parallel")
            serial = PipelineEngine(root / "serial")
            run(parallel, "events.jsonl", "features.json", "labels.json", reset=True)
            run(
                serial,
                "events.jsonl",
                "features.json",
                "labels.json",
                reset=True,
                config=SAMPLES / "config_serial.json",
            )
            parallel_state, _ = SnapshotStore(root / "parallel").recover()
            serial_state, _ = SnapshotStore(root / "serial").recover()
            self.assertTrue(compare_states(parallel_state, serial_state)["equal"])


if __name__ == "__main__":
    unittest.main()
