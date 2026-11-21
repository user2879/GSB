"""并发执行与单线程串行结果逐字段一致；预算约束确定性。"""
from __future__ import annotations

import unittest

from recon_workbench.artifact_store import ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.engine import WorkbenchEngine

from ._helpers import SampleHarness


def _run(threads: int, harness: SampleHarness, config: RebuildConfig,
         work_dir: str):
    load = harness.load()
    store = ArtifactStore(work_dir)
    engine = WorkbenchEngine(load, config, store, concurrency=threads)
    result = engine.run("float32", version=1)
    return result


class ConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = self.harness.config(thread_count=4)

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_serial_equals_parallel(self) -> None:
        serial = _run(1, self.harness, self.config,
                      self.harness.root + "/work_serial")
        parallel = _run(4, self.harness, self.config,
                        self.harness.root + "/work_parallel")

        self.assertEqual(
            [f.to_dict() for f in serial.features_flat],
            [f.to_dict() for f in parallel.features_flat],
        )
        self.assertEqual(
            [m.to_dict() for m in serial.matches_flat],
            [m.to_dict() for m in parallel.matches_flat],
        )
        self.assertEqual(
            [p.to_dict() for p in serial.tripoints_flat],
            [p.to_dict() for p in parallel.tripoints_flat],
        )
        self.assertEqual(
            [d.summary for d in serial.depth_maps],
            [d.summary for d in parallel.depth_maps],
        )
        self.assertEqual(
            [c.to_dict() for c in serial.cloud],
            [c.to_dict() for c in parallel.cloud],
        )
        self.assertEqual(
            serial.budget_deductions, parallel.budget_deductions
        )
        self.assertEqual(serial.failures, parallel.failures)

    def test_concurrency_capped_at_config(self) -> None:
        load = self.harness.load()
        store = ArtifactStore(self.harness.root + "/work_cap")
        engine = WorkbenchEngine(
            load, self.config, store, concurrency=99
        )
        self.assertLessEqual(engine.concurrency, self.config.thread_count)

    def test_budget_pending_is_marked(self) -> None:
        config = self.harness.config(rebuild_point_budget=30)
        serial = _run(1, self.harness, config,
                      self.harness.root + "/work_budget")
        pending_codes = {item["code"] for item in serial.pending}
        self.assertTrue(serial.budget_deductions)
        # 预算小于总需求时必须出现待算标记
        if any(
            d["granted"] < d["requested"] for d in serial.budget_deductions
        ):
            self.assertIn("E_REBUILD_BUDGET_PENDING", pending_codes)
        # 扣减序列严格按视图稳定顺序
        views = [d["view"] for d in serial.budget_deductions]
        self.assertEqual(views, sorted(views))


if __name__ == "__main__":
    unittest.main()
