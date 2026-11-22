from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_cli(self, *args: str) -> tuple[int, dict]:
        command = [
            sys.executable,
            "-m",
            "slowdiag",
            *args,
            "--workspace",
            str(self.workspace),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, json.loads(completed.stdout)

    def test_demo_and_three_injections(self) -> None:
        code, demo = self.run_cli("demo")
        self.assertEqual(code, 0)
        self.assertEqual(demo["business_code"], "OK")
        self.assertEqual(demo["status"], "demo_completed")
        self.assertTrue(demo["repeatability"]["repeatable"])
        self.assertEqual(demo["restore"]["snapshot_diff_count"], 0)

        code, result = self.run_cli("run", "--inject", "source")
        self.assertEqual(code, 2)
        self.assertEqual(result["business_code"], "E_SOURCE")
        self.assertEqual(result["path"], "sources[0].type")

        code, result = self.run_cli("run", "--inject", "format")
        self.assertEqual(code, 2)
        self.assertEqual(result["business_code"], "E_FORMAT")
        self.assertEqual(result["source_id"], "app-prod")
        self.assertGreaterEqual(result["row_no"], 1)

        code, result = self.run_cli("run", "--inject", "empty")
        self.assertEqual(code, 2)
        self.assertEqual(result["business_code"], "E_EMPTY")

        log_path = self.workspace / "logs" / "slowdiag.jsonl"
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        error_codes = {
            record.get("code") for record in records if record.get("event") == "command_error"
        }
        self.assertTrue({"E_SOURCE", "E_FORMAT", "E_EMPTY"}.issubset(error_codes))


if __name__ == "__main__":
    unittest.main()
