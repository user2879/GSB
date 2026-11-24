from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from matgen.cli import main

from tests.helpers import BASIC, INJECTED


class CliTests(unittest.TestCase):
    def test_basic_run_writes_verified_report_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(
                main(["run", "--input", str(BASIC), "--output", str(output), "--fresh"]),
                0,
            )
            report_path = output / "diagnostic_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["interactive"]["stale_responses"], 0)
            self.assertTrue(report["full_rebuild"]["exact_match"])
            self.assertEqual(report["error_trace"]["attribution"]["category"], "mix_order")
            self.assertEqual(report["scheduler"]["pending_code"], "E_POINT_BUDGET_PENDING")

            self.assertEqual(
                main(["run", "--input", str(BASIC), "--output", str(output), "--resume"]),
                0,
            )
            resumed = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(resumed["interactive"]["resume"]["usable"])
            self.assertEqual(resumed["interactive"]["commands_processed"], 0)
            self.assertTrue(resumed["full_rebuild"]["exact_match"])

    def test_injected_cli_reports_expected_codes_and_stale_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(
                main(
                    [
                        "run",
                        "--input",
                        str(INJECTED),
                        "--output",
                        str(output),
                        "--injections",
                        str(INJECTED / "injection_config.json"),
                        "--fresh",
                    ]
                ),
                0,
            )
            report = json.loads((output / "diagnostic_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "partial")
            self.assertTrue(report["injections"]["all_triggered_codes_matched"])
            self.assertEqual(report["interactive"]["stale_responses"], 6)
            self.assertIn("E_TEXTURE_HEADER", report["codes"])
            self.assertIn("E_MATERIAL_RUNTIME", report["codes"])


if __name__ == "__main__":
    unittest.main()
