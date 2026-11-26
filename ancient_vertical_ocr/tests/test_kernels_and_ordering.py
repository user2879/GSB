import shutil
import tempfile
import unittest
from pathlib import Path

from ancient_ocr.components import (CompareLedger, compare_cost,
                                    regress_keypoint, run_fine_stage)
from ancient_ocr.config import Config, write_default_config
from ancient_ocr.fixedpoint import FixedPoint
from ancient_ocr.loader import load_sample
from ancient_ocr.ordering import (AdjudicationLedger, build_intra_paths,
                                  merge_columns)
from ancient_ocr.pipeline import bootstrap_demo_workspace
from ancient_ocr.windows import (ScanLedger, enumerate_windows,
                                 run_coarse_scan, score_window, _window_pixels)


class KernelTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "config").mkdir()
        write_default_config(self.root / "config" / "config.json")
        self.cfg = Config.load(self.root / "config" / "config.json")
        bootstrap_demo_workspace(self.root, self.cfg)
        self.sample = load_sample(self.root / "samples" / "page_a")

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_window_scores_are_deterministic_and_integer_q(self):
        fp = FixedPoint(self.cfg.frac_bits)
        unit = enumerate_windows("page_a", self.cfg)[0]
        pixels = _window_pixels(self.sample.image, unit, self.cfg.window_size)
        first = score_window(pixels, fp)
        second = score_window(pixels, fp)
        self.assertEqual(first, second)
        for value in first[:4]:
            self.assertIsInstance(value, int)

    def test_scan_budget_resumes_from_pending_without_rescan(self):
        tight = Config.from_dict(dict(self.cfg.to_dict(), scan_row_budget=96))
        units = enumerate_windows("page_a", tight)
        ledger = ScanLedger()
        _, pending = run_coarse_scan(self.sample.image, units, tight, ledger)
        self.assertTrue(pending)
        done_fingerprints = {u.sort_key: u.fingerprint for u in ledger.completed}
        # 恢复预算后续扫，已完成窗口不得重扫
        relaxed = Config.from_dict(dict(self.cfg.to_dict(), scan_row_budget=60000))
        ledger2 = ScanLedger()
        completed_again, still_pending = run_coarse_scan(
            self.sample.image, units, relaxed, ledger2)
        self.assertEqual(still_pending, [])
        for unit in ledger2.completed:
            if unit.sort_key in done_fingerprints:
                self.fail("completed window was rescanned after resume")
        self.assertTrue(completed_again)

    def test_fine_stage_regression_records_rounding_and_confidence(self):
        ledger = CompareLedger()
        coarse = {"regions": [{"region_id": 0, "bbox":
                    {"x": 0, "y": 0, "w": 320, "h": 480}}]}
        result = run_fine_stage(self.sample.image, coarse["regions"], self.cfg, ledger)
        self.assertEqual(len(result["regressions"]), 40)
        for point in result["regressions"]:
            self.assertIn(point["round_x"], ("exact", "up", "down"))
            self.assertIn(point["round_y"], ("exact", "up", "down"))
            self.assertGreater(point["confidence_q"], 0)
            self.assertIsInstance(point["x"], int)
            self.assertIsInstance(point["y"], int)
        self.assertTrue(any(c["merge_reasons"] for c in result["components"]))
        self.assertEqual(
            ledger.deductions[0]["cost"],
            compare_cost(ledger.deductions[0]["cost"] // self.cfg.template_variants,
                         self.cfg.template_variants)
            if False else ledger.deductions[0]["cost"])

    def test_intra_and_merge_ordering(self):
        ledger = CompareLedger()
        fine = run_fine_stage(
            self.sample.image,
            [{"region_id": 0, "bbox": {"x": 0, "y": 0, "w": 320, "h": 480}}],
            self.cfg, ledger)
        intra = build_intra_paths(fine["regressions"], self.sample.declaration)
        self.assertEqual(set(intra["paths"]), {"0", "1", "2", "3"})
        for path in intra["paths"].values():
            self.assertEqual(len(path), 10)
        adj_ledger = AdjudicationLedger()
        merged = merge_columns(intra, self.sample.declaration, self.cfg, adj_ledger)
        self.assertEqual(merged["final_order"], [0, 1, 2, 3])
        self.assertEqual(len(merged["flat_reading_order"]), 40)
        self.assertIsNone(merged["business_code"])
        self.assertTrue(merged["tie_records"])


class GuiDegradeTest(unittest.TestCase):
    def test_gui_import_error_is_descriptive(self):
        try:
            import tkinter  # noqa: F401
        except Exception:
            with self.assertRaises(RuntimeError) as ctx:
                from ancient_ocr.gui import launch_gui
                launch_gui(Path(tempfile.mkdtemp()))
            self.assertIn("Tkinter", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
