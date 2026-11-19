from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .common import ROOT


class CliFaultTests(unittest.TestCase):
    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "bucketpack", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    def test_four_injection_codes_write_structured_logs(self) -> None:
        cases = {
            "width": "E_WIDTH",
            "bucket": "E_BUCKET",
            "cycle": "E_CYCLE",
            "ledger": "E_LEDGER",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for injection, expected in cases.items():
                with self.subTest(injection=injection):
                    ledger = root / f"{injection}.sqlite"
                    log = root / f"{injection}.jsonl"
                    result = self._run(
                        [
                            "run",
                            "--spec",
                            str(ROOT / "samples" / "spec.json"),
                            "--ledger",
                            str(ledger),
                            "--result-out",
                            str(root / f"{injection}_result.json"),
                            "--inject",
                            injection,
                            "--log-file",
                            str(log),
                        ]
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(result.stderr)
                    self.assertEqual(payload["code"], expected)
                    log_text = log.read_text(encoding="utf-8")
                    self.assertIn(expected, log_text)

    def test_forced_termination_then_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "crash.sqlite"
            result = self._run(
                [
                    "run",
                    "--spec",
                    str(ROOT / "samples" / "spec.json"),
                    "--ledger",
                    str(ledger),
                    "--batch-id",
                    "crash-cli",
                    "--result-out",
                    str(root / "result.json"),
                    "--crash-after",
                    "2",
                    "--log-file",
                    str(root / "run.jsonl"),
                ]
            )
            self.assertEqual(result.returncode, 17)
            recovery = self._run(
                [
                    "recover",
                    "--spec",
                    str(ROOT / "samples" / "spec.json"),
                    "--ledger",
                    str(ledger),
                    "--batch-id",
                    "crash-cli",
                    "--report-out",
                    str(root / "recovery.json"),
                    "--log-file",
                    str(root / "recover.jsonl"),
                ]
            )
            self.assertEqual(recovery.returncode, 0, recovery.stderr)
            payload = json.loads(recovery.stdout)
            self.assertEqual(payload["status"], "recovered")
            self.assertTrue(payload["rerun_packs"])
            self.assertEqual(payload["unknown_packs"], payload["rerun_packs"])


if __name__ == "__main__":
    unittest.main()
