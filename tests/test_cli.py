from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fixedseason.cli import main


class CliTests(unittest.TestCase):
    def test_generate_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = root / "fixtures"
            self.assertEqual(
                main(["generate", "--directory", str(fixtures), "--force"]), 0
            )
            workspace = root / "run"
            exit_code = main(
                [
                    "run",
                    "--series",
                    str(fixtures / "series.jsonl"),
                    "--config",
                    str(fixtures / "config.yaml"),
                    "--workspace",
                    str(workspace),
                    "--fresh",
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(
                (workspace / "final_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["run"]["complete"])
            self.assertEqual(report["summary"]["completed_series_count"], 8)
            self.assertTrue(
                (workspace / "snapshot" / "current" / "manifest.json").exists()
            )

    def test_invalid_config_still_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            series = root / "series.jsonl"
            config = root / "bad.yaml"
            workspace = root / "workspace"
            series.write_text(
                '{"series_id":"a","timestamps":[0,1,2],"values":[1,2,3]}\n',
                encoding="utf-8",
            )
            config.write_text(
                "model:\n  harmonic_order: 99\nfixed_point:\n  bits: 3\n",
                encoding="utf-8",
            )
            exit_code = main(
                [
                    "run",
                    "--series",
                    str(series),
                    "--config",
                    str(config),
                    "--workspace",
                    str(workspace),
                    "--fresh",
                ]
            )
            self.assertEqual(exit_code, 2)
            report = json.loads(
                (workspace / "final_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["validation_failures"])
            self.assertTrue(
                all(
                    issue["code"] == "E_CONFIG_INVALID"
                    for issue in report["validation_failures"]
                )
            )


if __name__ == "__main__":
    unittest.main()
