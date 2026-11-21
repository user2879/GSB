from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from candidate_exchange.injection import apply_injections

from tests.support import PROJECT_ROOT, ensure_sample_data


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ensure_sample_data()
        cls.environment = dict(os.environ)
        cls.environment["PYTHONPATH"] = (
            str(PROJECT_ROOT) + os.pathsep + cls.environment.get("PYTHONPATH", "")
        )

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "candidate_exchange", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=self.environment,
            timeout=120,
        )

    def test_injected_business_failures_keep_success_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            injected = root / "injected"
            apply_injections(self.data, injected, self.data / "injections.json", force=True)
            process = self._run(
                "run",
                "--data",
                str(injected),
                "--out",
                str(root / "out"),
                "--snapshots",
                str(root / "snapshots"),
                "--workers",
                "2",
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((root / "out" / "validation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed_with_failures")
            self.assertTrue(report["failures"])

    def test_simulated_termination_exit_code_has_no_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = self._run(
                "run",
                "--data",
                str(self.data),
                "--out",
                str(root / "out"),
                "--snapshots",
                str(root / "snapshots"),
                "--crash-after-stage",
                "matching",
            )
            self.assertEqual(process.returncode, 75)
            self.assertIn("SIMULATED_TERMINATION", process.stderr)
            self.assertFalse((root / "out" / "validation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
