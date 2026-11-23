"""阶段化快照整体提交与 E_SNAPSHOT_STALE 判定。"""
import os
import tempfile
import unittest

from reward_workbench.config import load_config
from reward_workbench.errors import E_SNAPSHOT_STALE, FailureLog
from reward_workbench.pairs import load_pairs
from reward_workbench.pipeline import Pipeline
from reward_workbench.samplegen import write_workspace
from reward_workbench.snapshot import SnapshotStore


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rwb-snap-")
        self.paths = write_workspace(self.tmp)
        self.snap = os.path.join(self.tmp, "runtime", "snapshots")
        self.tmpd = os.path.join(self.tmp, "runtime", "tmp")

    def _load(self):
        fl = FailureLog()
        cfg = load_config(self.paths["config"], fl)
        pairs = load_pairs(self.paths["config"] and self.paths["pairs"],
                           cfg.normalize, fl)
        return cfg, pairs

    def test_commit_and_manifest(self):
        cfg, pairs = self._load()
        store = SnapshotStore(self.snap, self.tmpd)
        pipe = Pipeline(cfg, pairs, snapshot_store=store)
        pipe.run_all_batches()
        version = store.latest_version()
        self.assertIsNotNone(version)
        manifest = store.load_manifest(version)
        self.assertEqual(manifest.consumed_budget, pipe.budget.consumed)
        self.assertTrue(os.path.exists(
            os.path.join(self.snap, f"v{version}", "summary.json")))

    def test_stale_on_config_change(self):
        cfg, pairs = self._load()
        store = SnapshotStore(self.snap, self.tmpd)
        pipe = Pipeline(cfg, pairs, snapshot_store=store)
        pipe.run_all_batches()

        # 用不同配置哈希做启动判定 -> E_SNAPSHOT_STALE
        fl = FailureLog()
        manifest, is_stale = store.resume_or_detect_stale(
            expected_config_hash="different",
            expected_data_fp="different",
            failures=fl,
        )
        self.assertTrue(is_stale)
        self.assertTrue(fl.has(E_SNAPSHOT_STALE))

    def test_resume_when_matching(self):
        cfg, pairs = self._load()
        store = SnapshotStore(self.snap, self.tmpd)
        pipe = Pipeline(cfg, pairs, snapshot_store=store)
        pipe.run_all_batches()
        fl = FailureLog()
        manifest, is_stale = store.resume_or_detect_stale(
            expected_config_hash=pipe._config_hash(),
            expected_data_fp=pipe._data_fingerprint(),
            failures=fl,
        )
        self.assertFalse(is_stale)
        self.assertIsNotNone(manifest)


if __name__ == "__main__":
    unittest.main()
