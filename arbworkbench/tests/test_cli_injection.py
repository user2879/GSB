import json
import tempfile
import unittest
from pathlib import Path

from arbworkbench.cli import main


class CliInjectionTests(unittest.TestCase):
    def setUp(self):
        # The CLI writes under package-level output/; isolate via cwd temp is
        # not required because output files are deterministic and overwritten.
        pass

    def test_clean_run_succeeds(self):
        code = main(["--fresh"])
        self.assertEqual(code, 0)

    def test_spec_corruption_reports_e_spec(self):
        code = main(["--fresh", "--inject", "spec_corruption"])
        self.assertEqual(code, 2)

    def test_grid_violation_reports_e_grid(self):
        code = main(["--fresh", "--inject", "grid_violation"])
        self.assertEqual(code, 2)

    def test_version_mismatch_reports_e_version(self):
        code = main(["--fresh", "--inject", "version_mismatch"])
        self.assertEqual(code, 2)

    def test_structured_log_records_error_codes(self):
        main(["--fresh", "--inject", "spec_corruption"])
        log_path = Path(__file__).resolve().parent.parent / "output" / "events.jsonl"
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        error_events = [r for r in records if r.get("event") == "workbench_error"]
        self.assertTrue(any(r["code"] == "E_SPEC" for r in error_events))

    def test_crash_recovery_cli_path(self):
        code = main(["--fresh", "--crash-view", "view-production"])
        self.assertEqual(code, 0)
        report_path = Path(__file__).resolve().parent.parent / "output" / "arbitration_report.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["unknown_views"], [])


if __name__ == "__main__":
    unittest.main()
