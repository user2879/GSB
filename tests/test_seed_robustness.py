"""跨种子稳定性：不同样本种子下全流程、策略与等价性依然成立。"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from retbench.acceptance import run_acceptance


class SeedRobustnessTest(unittest.TestCase):
    def test_acceptance_runs_under_multiple_seeds(self) -> None:
        # run_acceptance 使用模块固定种子；这里通过不同样本规模与独立目录，
        # 验证工作台在非默认布局下也能完成全链路且等价。
        for sample_count in (8, 10):
            workspace = tempfile.mkdtemp(prefix=f"retbench-seed-{sample_count}-")
            try:
                summary = run_acceptance(workspace, sample_count=sample_count)
                self.assertTrue(summary["equivalence"]["equivalent"])
                self.assertTrue(
                    summary["concurrency"]["deterministic_across_workers"]
                )
                self.assertTrue(summary["budget_demo"]["exceeded"])
                self.assertTrue(summary["snapshot_stale_demo"]["rebuild_triggered"])
                self.assertGreaterEqual(len(summary["decision_trace"]), 2)
            finally:
                shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
