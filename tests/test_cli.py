from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliInjectionTests(unittest.TestCase):
    def _run(self, injection: str) -> tuple[int, dict, list[dict]]:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "slowquery_workbench",
                    "demo",
                    "--work-dir",
                    directory,
                    "--inject",
                    injection,
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            payload = json.loads(completed.stdout)
            log_path = Path(payload["log"])
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            return completed.returncode, payload, records

    def test_data_injection_is_genuine_e_data(self) -> None:
        code, payload, records = self._run("data")
        self.assertEqual(code, 2)
        self.assertEqual(payload["code"], "E_DATA")
        self.assertIn("NOT NULL constraint failed", payload["message"])
        self.assertTrue(any(item.get("code") == "E_DATA" for item in records))

    def test_index_version_injection_is_e_version(self) -> None:
        code, payload, records = self._run("index-version")
        self.assertEqual(code, 2)
        self.assertEqual(payload["code"], "E_VERSION")
        self.assertEqual(payload["details"]["differences"][0]["field"], "index_version")
        self.assertTrue(any(item.get("code") == "E_VERSION" for item in records))

    def test_timeout_injection_uses_monotonic_timing(self) -> None:
        code, payload, records = self._run("timeout")
        self.assertEqual(code, 2)
        self.assertEqual(payload["code"], "E_TIMEOUT")
        self.assertGreaterEqual(payload["details"]["elapsed_ms"], payload["details"]["timeout_ms"])
        self.assertTrue(any(item.get("code") == "E_TIMEOUT" for item in records))

    def test_normal_demo_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "slowquery_workbench",
                    "demo",
                    "--work-dir",
                    directory,
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["candidate_kinds"]), 4)
        self.assertTrue(payload["stale_after_index_change"])
        self.assertEqual(payload["snapshot_restore_version_error"]["code"], "E_VERSION")
        self.assertTrue(payload["replay_matched"])


if __name__ == "__main__":
    unittest.main()
