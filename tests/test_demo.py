from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cacheproof.demo import run_demo


class DemoTest(unittest.TestCase):
    def test_demo_is_repeatable_and_records_required_codes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheproof-demo-") as temp:
            workspace = Path(temp) / "demo"
            report_path = Path(temp) / "report.json"
            first = run_demo(workspace, reset=True, report_path=report_path)
            self.assertTrue(first["ok"])
            self.assertTrue(report_path.exists())
            phases = {phase["phase"]: phase for phase in first["phases"]}
            self.assertEqual(phases["capacity_rejection"]["code"], "E_CAPACITY")
            self.assertTrue(phases["capacity_rejection"]["proof_unchanged"])
            self.assertEqual(phases["crash_recovery"]["worker_exit_code"], 91)
            self.assertTrue(phases["crash_recovery"]["proofs_still_ok"])
            self.assertFalse(phases["crash_recovery"]["exit_report_available"])
            self.assertEqual(
                [item["entry_key"] for item in phases["crash_recovery"]["unknown_entries"]],
                ["alpha/crash"],
            )
            injection_codes = {
                case["injection"]: case["expected_code"]
                for case in phases["structured_injections"]["cases"]
            }
            self.assertEqual(
                injection_codes,
                {
                    "truncate": "E_INTEGRITY",
                    "capacity": "E_CAPACITY",
                    "ledger-digest": "E_SNAPSHOT",
                },
            )
            second = run_demo(workspace, reset=True)
            second_phases = {phase["phase"]: phase for phase in second["phases"]}
            self.assertEqual(
                phases["initialize"]["segments"], second_phases["initialize"]["segments"]
            )
            self.assertEqual(
                phases["view_rebuild"]["views"],
                second_phases["view_rebuild"]["views"],
            )


if __name__ == "__main__":
    unittest.main()
