from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evidence_diag.cli import run_demo


class DemoIntegrationTests(unittest.TestCase):
    def test_end_to_end_demo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary = run_demo(Path(temp))
            checks = summary["checks"]
            self.assertTrue(checks["all_cursor_results_equal"])
            self.assertEqual(checks["cursor_count"], 18)
            self.assertEqual(checks["budget_pending_count"], 7)
            self.assertEqual(checks["failure_codes"], ["E_INPUT_INVALID"])
            self.assertTrue(checks["fixed_point_change_verified"])
            self.assertGreater(checks["diagnostic_first_exceed_count"], 0)
            self.assertGreater(checks["diagnostic_flip_count"], 0)
            self.assertTrue(checks["snapshot_crash_recovered"])
            self.assertTrue(checks["config_mismatch_full_recompute"])
            reports = Path(temp) / "reports"
            for name in (
                "equivalence.json",
                "precision.json",
                "diagnostic.json",
                "recovery.json",
                "demo_summary.json",
            ):
                self.assertTrue((reports / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
