"""阶段快照：原子整体替换、续跑判定与 E_SNAPSHOT_MISMATCH 恢复。"""
from __future__ import annotations

import unittest

from recon_workbench.artifact_store import STAGES, ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.engine import WorkbenchEngine
from recon_workbench.errors import BizCode, WorkbenchError
from recon_workbench.loader import input_fingerprint, load_sample

from ._helpers import SampleHarness


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = RebuildConfig()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_all_stages_committed(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=1)
        engine.run("float64", version=1)
        for stage in STAGES:
            manifest = store.stage_manifest(1, stage)
            self.assertIsNotNone(manifest, stage)
            self.assertEqual(manifest["config_hash"],
                             self.config.fingerprint())
            self.assertEqual(manifest["input_hash"],
                             input_fingerprint(load.views))

    def test_resume_with_identical_input(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=1)
        engine.run("float64", version=1)
        manifests = store.verify_resume(
            1, self.config.fingerprint(), input_fingerprint(load.views)
        )
        self.assertEqual(set(manifests), set(STAGES))

    def test_mismatch_triggers_discard_and_replace(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=1)
        engine.run("float64", version=1)
        with self.assertRaises(WorkbenchError) as ctx:
            store.verify_resume(1, "deadbeefdeadbeef",
                                input_fingerprint(load.views))
        self.assertEqual(ctx.exception.code, BizCode.E_SNAPSHOT_MISMATCH)

        # 模拟运行器恢复路径：废弃不匹配版本并整体替换
        store.discard_version(1)
        self.assertIsNone(store.latest_version())
        engine.run("float64", version=1)
        self.assertEqual(store.latest_version(), 1)

    def test_snapshot_manifest_has_required_fields(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=1)
        engine.run("float64", version=1)
        manifest = store.stage_manifest(1, "point_cloud")
        for field in ("version", "stage", "config_hash", "input_hash",
                      "precision", "completed_views"):
            self.assertIn(field, manifest)


if __name__ == "__main__":
    unittest.main()
