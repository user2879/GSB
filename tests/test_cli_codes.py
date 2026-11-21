from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliBusinessCodeTests(unittest.TestCase):
    def run_injection(self, kind: str, expected_exit: int, expected_code: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo_root = Path(directory) / "demo"
            command = [
                sys.executable,
                "-m",
                "streamagg.cli",
                "inject",
                "--kind",
                kind,
                "--workspace",
                str(ROOT),
                "--demo-root",
                str(demo_root),
                "--ledger",
                str(demo_root / "ledger.sqlite3"),
                "--snapshots",
                str(demo_root / "snapshots"),
                "--logs",
                str(demo_root / "logs"),
                "--reports",
                str(demo_root / "reports"),
            ]
            process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(process.returncode, expected_exit, process.stderr)
            payload = json.loads(process.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["code"], expected_code)
            log_path = demo_root / "logs" / f"injection_{kind}.jsonl"
            self.assertTrue(log_path.exists())
            log_lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(line["code"] == expected_code for line in log_lines))

    def test_order_injection_code(self) -> None:
        self.run_injection("order", 3, "E_ORDER")

    def test_grid_injection_code(self) -> None:
        self.run_injection("grid", 4, "E_GRID")

    def test_snapshot_injection_code(self) -> None:
        self.run_injection("snapshot", 5, "E_SNAPSHOT")

    def test_replay_injection_code(self) -> None:
        self.run_injection("replay", 6, "E_REPLAY")


if __name__ == "__main__":
    unittest.main()
