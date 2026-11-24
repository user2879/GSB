from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT, make_small_sources


class AnomalyCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="anomaly-test-")
        cls.directory = Path(cls.temp.name)
        cls.sources, cls.rules, cls.budgets = make_small_sources(
            cls.directory, event_count=40
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def run_injection(self, mode: str, expected_code: str, expected_exit: int) -> None:
        ledger = self.directory / f"inject-{mode}.sqlite"
        log_path = self.directory / f"inject-{mode}.jsonl"
        command = [
            sys.executable,
            "-m",
            "lateness_ledger",
            "run",
            "--sources",
            str(self.sources[0]),
            "--rules",
            str(self.rules),
            "--budgets",
            str(self.budgets),
            "--ledger",
            str(ledger),
            "--inject",
            mode,
            "--log-json",
            str(log_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, expected_exit, completed.stderr)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], expected_code)
        self.assertTrue(payload["error"]["details"]["injected"])
        self.assertTrue(log_path.exists())
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(record["code"] == expected_code for record in records))
        self.assertIn("event_id", payload["error"]["details"])
        if mode == "budget":
            self.assertIn("window_id", payload["error"]["details"])

    def test_order_injection_reports_e_order(self) -> None:
        self.run_injection("order", "E_ORDER", 3)

    def test_budget_injection_reports_e_budget(self) -> None:
        self.run_injection("budget", "E_BUDGET", 4)

    def test_watermark_injection_reports_e_watermark(self) -> None:
        self.run_injection("watermark", "E_WATERMARK", 5)


if __name__ == "__main__":
    unittest.main()
