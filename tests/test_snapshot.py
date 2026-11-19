import json
import tempfile
import unittest
from pathlib import Path

from segpos.errors import ConfigMismatchError
from tests._fixtures import make_pipeline, write_workspace


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = write_workspace(self.root / "ws", workers=1, interval=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_reuses_committed_snapshot(self):
        snap = self.root / "snap"
        first = make_pipeline(self.paths, snap)
        outcome = first.run()
        self.assertGreaterEqual(outcome.report["snapshot_version"], 1)
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["completed_sentences"]), 5)
        self.assertEqual(
            sorted(manifest["artifact_versions"]),
            ["oov", "segmentation", "summary", "tagging"],
        )

        second = make_pipeline(self.paths, self.root / "snap2")
        second.snapshot = first.snapshot
        outcome2 = second.run()
        self.assertEqual(len(outcome2.reused), 5)
        self.assertEqual(outcome2.recomputed, [])
        self.assertEqual(
            second.results_by_id()["S1"]["tokens"],
            first.results_by_id()["S1"]["tokens"],
        )

    def test_resume_mismatch_rejects_with_e_config_mismatch(self):
        snap = self.root / "snap"
        first = make_pipeline(self.paths, snap)
        first.run()

        dict_path = self.paths["dictionary"]
        dict_path.write_text(
            dict_path.read_text(encoding="utf-8") + "新词 12 n\n",
            encoding="utf-8",
        )
        second = make_pipeline(self.paths, self.root / "other")
        second.snapshot = first.snapshot
        with self.assertRaises(ConfigMismatchError) as ctx:
            second.run()
        self.assertEqual(ctx.exception.code, "E_CONFIG_MISMATCH")

    def test_report_written_atomically_and_contains_hashes(self):
        snap = self.root / "snap"
        pipe = make_pipeline(self.paths, snap)
        report_path = self.root / "report.json"
        outcome = pipe.run(report_path)
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dict_hash"], pipe.dictionary.hash)
        self.assertEqual(payload["prob_table_hash"], pipe.prob_table.hash)
        self.assertEqual(payload["snapshot_version"], outcome.report["snapshot_version"])
        self.assertEqual(payload["total_sentences"], 5)
        self.assertIn("failures", payload)

    def test_multi_batch_commit_keeps_all_sentences(self):
        paths = write_workspace(self.root / "ws_multi", workers=1, interval=2)
        snap = self.root / "snap_multi"
        pipe = make_pipeline(paths, snap)
        pipe.run()
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(manifest["completed_sentences"]),
            ["S1", "S2", "S3", "S4", "S5"],
        )
        result_files = list((snap / "results").glob("*.json"))
        self.assertEqual(len(result_files), 5)
        self.assertGreaterEqual(manifest["global_version"], 2)


if __name__ == "__main__":
    unittest.main()
