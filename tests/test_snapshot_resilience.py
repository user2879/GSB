import json
import os
import shutil
import tempfile
import unittest

from ab_workbench.config import load_config
from ab_workbench.generator import generate
from ab_workbench.pipeline import GracefulStop, Workbench


class SnapshotResilienceTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.sample = os.path.join(self.work, "sample")
        generate(self.sample, num_users=30, requests_per_user=2, candidates_per_request=4)
        self.log = os.path.join(self.sample, "exposures.jsonl")
        self.attrs = os.path.join(self.sample, "user_attributes.json")
        self.cfg_path = os.path.join(self.sample, "experiment_config.json")

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _bench(self, tag):
        cfg = load_config(self.cfg_path)
        return Workbench(
            config=cfg,
            snapshot_dir=os.path.join(self.work, tag, "sn"),
            cache_dir=os.path.join(self.work, tag, "cache"),
            tmp_dir=os.path.join(self.work, tag, "tmp"),
            concurrent=True,
        )

    def test_graceful_stop_exit_code_zero(self):
        bench = self._bench("graceful")
        graceful = GracefulStop()
        graceful.requested = True
        graceful.signal_name = "SIGTERM"
        result = bench.run(self.log, self.attrs, graceful=graceful)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.signal_handled, "SIGTERM")
        manifest_path = os.path.join(self.work, "graceful", "sn", "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

    def test_recovery_uses_only_complete_snapshot(self):
        bench = self._bench("recovery")
        result = bench.run(self.log, self.attrs).to_dict()
        with open(os.path.join(self.work, "recovery", "sn", "manifest.json")) as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["global_version"], 1)
        # Every stage listed in the manifest must have a complete latest.json.
        for stage, version in manifest["stage_versions"].items():
            latest = os.path.join(self.work, "recovery", "sn", stage, "latest.json")
            self.assertTrue(os.path.exists(latest))
            with open(
                os.path.join(self.work, "recovery", "sn", stage, "version.txt")
            ) as handle:
                pointer = handle.read()
            self.assertEqual(pointer, version)
        # Resume after simulated hard kill reuses the persisted artifacts.
        again = bench.run(self.log, self.attrs).to_dict()
        self.assertTrue(again["cache"]["reused_keys"])
        self.assertEqual(again["group_stats"], result["group_stats"])


if __name__ == "__main__":
    unittest.main()
