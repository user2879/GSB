from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quantcalib.cli import main
from quantcalib.samplegen import generate_samples


class CliTests(unittest.TestCase):
    def test_run_and_receipt_conversion_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = generate_samples(root / "samples")
            report_path = root / "report.json"
            exit_code = main(
                [
                    "run",
                    "--weights",
                    str(samples["weights"]),
                    "--calibration",
                    str(samples["calibration"]),
                    "--config",
                    str(samples["config"]),
                    "--receipts",
                    str(samples["q1"]),
                    str(samples["q2"]),
                    str(samples["q3"]),
                    "--state",
                    str(root / "state"),
                    "--report",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "ok")
            exit_code = main(
                [
                    "run",
                    "--weights",
                    str(samples["weights"]),
                    "--calibration",
                    str(samples["calibration"]),
                    "--config",
                    str(samples["config"]),
                    "--state",
                    str(root / "default-report-state"),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "default-report-state" / "latest-report.json").exists())
            conversion_report = root / "conversion.json"
            exit_code = main(
                [
                    "convert-receipts",
                    "--input",
                    str(samples["q3"]),
                    "--output",
                    str(root / "q1.jsonl"),
                    "--to",
                    "q1",
                    "--sidecar",
                    str(root / "sidecar.json"),
                    "--report",
                    str(conversion_report),
                ]
            )
            self.assertEqual(exit_code, 0)
            converted = json.loads(conversion_report.read_text())
            self.assertTrue(converted["q3_bypass_fields_preserved"])


if __name__ == "__main__":
    unittest.main()
