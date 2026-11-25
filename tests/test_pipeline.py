from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vertical_ocr.diagnostics import _truth_accuracy, prepare_workspace
from vertical_ocr.pipeline import Pipeline


class PipelineAcceptanceTests(unittest.TestCase):
    def test_full_run_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ws"
            cfg = prepare_workspace(root)
            first = Pipeline(root, cfg).run()
            self.assertEqual(first["receipt_counts"]["generated_records"], 54)
            self.assertEqual(first["failures"], [])
            accuracy = _truth_accuracy(root)
            self.assertEqual((accuracy["correct"], accuracy["total"]), (54, 54))

            second = Pipeline(root, cfg).run()
            self.assertEqual(second["cache_counts"]["recomputed"], 0)
            self.assertEqual(second["cache_counts"]["hit"], 123)
            self.assertTrue(second["snapshot"]["fresh"])

    def test_budget_pending_then_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ws"
            cfg = prepare_workspace(root)
            first = Pipeline(root, cfg).run(batch_budget=70_000)
            self.assertGreater(first["budget"]["pending_count"], 0)
            second = Pipeline(root, cfg).run(batch_budget=10_000_000)
            self.assertEqual(second["budget"]["pending_count"], 0)


if __name__ == "__main__":
    unittest.main()
