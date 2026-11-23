import tempfile
import unittest
from pathlib import Path

from tests._util import make_workspace, run
from mstc.errors import Code
from mstc.scheduler import task_sort_key


class SchedulerTests(unittest.TestCase):
    def test_stable_sort_key_ordering(self):
        keys = [
            task_sort_key("classify", 1, 2, "t12"),
            task_sort_key("classify", 0, 0, "t00"),
            task_sort_key("classify", 0, 1, "t01"),
            task_sort_key("fuse", 0, 0, "t00"),
        ]
        self.assertEqual(sorted(keys), [
            ("classify", 0, 0, "t00"),
            ("classify", 0, 1, "t01"),
            ("classify", 1, 2, "t12"),
            ("fuse", 0, 0, "t00"),
        ])

    def test_global_pixel_budget(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        cfg = ws / "config.yaml"
        cfg.write_text(cfg.read_text().replace(
            "global_pixel_budget: 100000", "global_pixel_budget: 300"))
        report, _ = run(ws)
        self.assertIn(Code.E_PIXEL_BUDGET_EXHAUSTED, report.failures.codes())

    def test_tile_quota_budget(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        cfg = ws / "config.yaml"
        cfg.write_text(cfg.read_text().replace(
            "tile_pixel_quota: 1000", "tile_pixel_quota: 100"))
        report, _ = run(ws)
        self.assertIn(Code.E_TILE_BUDGET_EXHAUSTED, report.failures.codes())

    def test_fusion_batch_limit_pending(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        cfg = ws / "config.yaml"
        cfg.write_text(cfg.read_text().replace(
            "fusion_batch_limit: 32", "fusion_batch_limit: 3"))
        report, _ = run(ws)
        self.assertIn(Code.E_FUSION_BUDGET_PENDING, report.failures.codes())
        self.assertEqual(len(report.pending_fusions), 9)

    def test_concurrent_matches_serial(self):
        ws_par = make_workspace(Path(tempfile.mkdtemp()))
        ws_ser = make_workspace(Path(tempfile.mkdtemp()))
        (ws_ser / "config.yaml").write_text(
            (ws_ser / "config.yaml").read_text().replace("max_workers: 4", "max_workers: 1"))
        par, _ = run(ws_par)
        ser, _ = run(ws_ser)
        import numpy as np
        np.testing.assert_array_equal(
            par.global_raster.raster, ser.global_raster.raster)


if __name__ == "__main__":
    unittest.main()
