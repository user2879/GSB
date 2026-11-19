from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path = ROOT):
    return subprocess.run([sys.executable, "-m", "vecflow", *args], cwd=cwd, capture_output=True, text=True, timeout=40)


class CliTests(unittest.TestCase):
    def test_preflight_and_log_dir_writable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _run("preflight", "--log-dir", str(Path(td) / "logs"))
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["cli_parse"])
            self.assertTrue(payload["log_dir_writable"])
            self.assertTrue((Path(td) / "logs" / "vecflow.jsonl").exists())

    def test_injection_business_codes_are_logged(self) -> None:
        expected = {
            "range": "E_RANGE",
            "nullmap": "E_NULLMAP",
        }
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs"
            for kind, code in expected.items():
                with self.subTest(kind=kind):
                    result = _run("validate", f"--inject-{kind}", "--log-dir", str(log_dir))
                    self.assertEqual(result.returncode, 2)
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload["code"], code)
            budget = _run(
                "run",
                "--inject-budget",
                "--ledger",
                str(Path(td) / "budget.sqlite"),
                "--snapshot-dir",
                str(Path(td) / "snap"),
                "--log-dir",
                str(log_dir),
                "--reset-state",
            )
            self.assertEqual(budget.returncode, 3)
            self.assertEqual(json.loads(budget.stdout)["code"], "E_BUDGET")
            log_text = (log_dir / "vecflow.jsonl").read_text(encoding="utf-8")
            for code in ("E_RANGE", "E_NULLMAP", "E_BUDGET"):
                self.assertIn(f'"code":"{code}"', log_text)
            for line in log_text.splitlines():
                json.loads(line)


if __name__ == "__main__":
    unittest.main()
