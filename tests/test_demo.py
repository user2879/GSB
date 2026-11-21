from __future__ import annotations

import unittest

from mlpipe_workbench.demo import run_demo

from tests.helpers import TempDirectoryMixin


class DemoIntegrationTests(TempDirectoryMixin, unittest.TestCase):
    def test_three_scenarios_run_in_stable_order(self) -> None:
        summary = run_demo(self.tempdir / "demo")
        lines = summary.splitlines()
        scenario_lines = [line for line in lines if line.startswith("SCENARIO ")]
        self.assertEqual(
            scenario_lines,
            [
                "SCENARIO 1 COLD_START",
                "SCENARIO 2 SPEC_EVOLUTION",
                "SCENARIO 3 CRASH_RECOVERY",
            ],
        )
        self.assertIn("invalidated_shards=1", lines)
        self.assertIn("preserved_shards=4", lines)
        self.assertIn("crash_exit_code=75", lines)
        self.assertIn("exit_report_available=false", lines)
        self.assertIn("replay_status=OK", lines)
        self.assertIn("matches_cold_final=true", lines)
        self.assertIn("status_after_recovery=RECOVERED_COMPLETE", lines)
        self.assertTrue((self.tempdir / "demo" / "demo_summary.txt").exists())
        self.assertTrue((self.tempdir / "demo" / "demo_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
