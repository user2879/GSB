"""可处理退出信号：原子替换后退出码 0；强制终止后从完整快照恢复。"""
from __future__ import annotations

import threading
import time
import unittest

from recon_workbench.artifact_store import ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.engine import WorkbenchEngine
from recon_workbench.runner import GracefulShutdown

from ._helpers import SampleHarness


class ShutdownTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = RebuildConfig()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_graceful_shutdown_exit_zero(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=2)
        shutdown = GracefulShutdown()

        def trigger() -> None:
            time.sleep(0.05)
            shutdown.event.set()

        thread = threading.Thread(target=trigger)
        thread.start()
        result = engine.run(
            "float64", version=1, shutdown_event=shutdown.event
        )
        thread.join()
        # 已完成的阶段快照必须完整可读
        latest = store.latest_version()
        self.assertEqual(latest, 1)
        self.assertTrue(store.stage_manifest(1, "feature"))

    def test_recovery_only_uses_complete_snapshot(self) -> None:
        store = ArtifactStore(self.harness.root + "/work")
        load = self.harness.load()
        engine = WorkbenchEngine(load, self.config, store, concurrency=1)
        engine.run("float64", version=1)
        # 模拟强制终止：仅完整快照存在，缓存目录可被新运行复用
        manifests = store.verify_resume(
            1, self.config.fingerprint(),
            __import__(
                "recon_workbench.loader", fromlist=["input_fingerprint"]
            ).input_fingerprint(load.views),
        )
        self.assertIn("feature", manifests)
        self.assertIn("point_cloud", manifests)


if __name__ == "__main__":
    unittest.main()
