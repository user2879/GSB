import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from feature_pipeline.cache import IncrementalCache
from feature_pipeline.config import load_merge_config
from feature_pipeline.pipeline import Pipeline
from feature_pipeline.snapshot import SnapshotStore


REPO_ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name) / "work"
        cls.pipeline = Pipeline(
            sample_dir=REPO_ROOT / "samples",
            config_path=REPO_ROOT / "samples" / "merge_config.json",
            metadata_path=REPO_ROOT / "samples" / "feature_metadata.csv",
            work_dir=cls.work,
        )
        cls.summary = cls.pipeline.run()
        cls.snapshot = Path(cls.summary["pointer"]["snapshot_dir"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_manifest_and_required_artifacts(self) -> None:
        manifest = json.loads((self.snapshot / "manifest.json").read_text())
        self.assertEqual(manifest["global_version"], 1)
        self.assertIn("config_hash", manifest)
        self.assertIn("metadata_hash", manifest)
        self.assertEqual(set(manifest["source_fingerprints"]),
                         {"v1", "v2", "v3"})
        for name in (
            "merged_table.json",
            "conflicts.json",
            "interop_report.json",
            "failures.json",
            "incremental_audit.json",
        ):
            self.assertTrue((self.snapshot / name).exists())

    def test_interop_report_contract(self) -> None:
        report = json.loads((self.snapshot / "interop_report.json").read_text())
        self.assertEqual(report["report_type"], "cross_version_interop")
        self.assertTrue(report["cross_version_consistent"])
        self.assertEqual(report["consistency_judgement"], "CONSISTENT")
        for feature, entry in report["per_feature_deviation"].items():
            self.assertIn("deviation_histogram", entry)
            self.assertEqual(
                sum(entry["deviation_histogram"].values()),
                entry["sample_count"],
            )
        self.assertIsNotNone(report["max_deviation_feature_entity"]["feature"])
        self.assertIn("downgrade_losses", report)
        self.assertIn("missing_position_differences", report)

    def test_second_run_reuses_everything(self) -> None:
        second = Pipeline(
            sample_dir=REPO_ROOT / "samples",
            config_path=REPO_ROOT / "samples" / "merge_config.json",
            metadata_path=REPO_ROOT / "samples" / "feature_metadata.csv",
            work_dir=self.work,
        ).run()
        self.assertEqual(second["global_version"], 2)
        self.assertGreater(second["cache_audit"]["reused_count"], 0)
        self.assertEqual(second["cache_audit"]["recomputed_count"], 0)

    def test_incremental_invalidation_is_scoped(self) -> None:
        sample_copy = Path(self.tmp.name) / "samples_copy"
        sample_copy.mkdir()
        for name in ("features_v1.txt", "features_v3.bin",
                     "feature_metadata.csv", "merge_config.json"):
            (sample_copy / name).write_bytes(
                (REPO_ROOT / "samples" / name).read_bytes()
            )
        v2 = json.loads((REPO_ROOT / "samples" / "features_v2.json").read_text())
        v2["entities"]["ent_0005"]["f_income"]["value"] = 55.5
        (sample_copy / "features_v2.json").write_text(json.dumps(v2))

        work = Path(self.tmp.name) / "scoped_work"
        first = Pipeline(sample_copy, sample_copy / "merge_config.json",
                         sample_copy / "feature_metadata.csv", work).run()
        second = Pipeline(sample_copy, sample_copy / "merge_config.json",
                          sample_copy / "feature_metadata.csv", work).run()
        self.assertEqual(second["cache_audit"]["recomputed_count"], 0)

        # Mutate exactly one cell and rerun.
        v2["entities"]["ent_0005"]["f_income"]["value"] = 60.25
        (sample_copy / "features_v2.json").write_text(json.dumps(v2))
        third = Pipeline(sample_copy, sample_copy / "merge_config.json",
                         sample_copy / "feature_metadata.csv", work).run()
        invalidated = third["cache_audit"]["invalidated"]
        coordinates = {item["coordinate"] for item in invalidated}
        self.assertIn("ent_0005|f_income|v2", coordinates)
        self.assertIn("ent_0005|f_income", coordinates)
        self.assertNotIn("ent_0006|f_income|v2", coordinates)

    def test_recovery_point_requires_committed_snapshot(self) -> None:
        store = SnapshotStore(self.work / "snapshot_store")
        recovery = store.recovery_point()
        self.assertTrue(recovery["recoverable"])
        self.assertEqual(recovery["global_version"], 1)

    def test_graceful_signal_exits_zero_after_batch(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "signal_demo.py"),
                "--work-dir",
                str(Path(self.tmp.name) / "signal_work"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, msg=stderr)
        self.assertIn("SIGTERM", stdout)
        self.assertIn("graceful", stdout)


if __name__ == "__main__":
    unittest.main()
