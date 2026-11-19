from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spectral_workspace.demo import run_demo


class DemoDeterminismAndRecoveryTests(unittest.TestCase):
    def test_two_identical_demo_runs_are_byte_identical_and_recover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_a = root / "report_a.json"
            report_b = root / "report_b.json"
            first = run_demo(root / "run_a", report_a, force=True)
            second = run_demo(root / "run_b", report_b, force=True)

            self.assertEqual(report_a.read_bytes(), report_b.read_bytes())
            self.assertEqual(first["recovery_comparison"]["status"], "ok")
            self.assertTrue(first["acceptance"]["forced_recovery_colors_match_baseline"])
            self.assertTrue(
                first["acceptance"]["forced_recovery_stage_hashes_match_baseline"]
            )
            self.assertTrue(first["acceptance"]["uncommitted_transaction_was_orphaned"])
            self.assertEqual(first["workspace"]["revision"], 6)
            self.assertEqual(first["workspace"]["snapshot_version"], 6)
            self.assertEqual(
                first["recovery"]["prior_exit_state"],
                "forced_termination_recovered",
            )
            self.assertFalse(first["recovery"]["exit_report_fabricated"])
            self.assertEqual(
                first["acceptance"]["injection_error_codes"],
                {
                    "invalid_injection_config": "E_INJECT_CONFIG",
                    "observer_invalid": "E_OBSERVER_INVALID",
                    "reflectance_out_of_range": "E_SPECTRUM_RANGE",
                    "stage_failure": "E_STAGE_STALE",
                    "white_point_zero": "E_WHITE_POINT",
                },
            )


if __name__ == "__main__":
    unittest.main()
