from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DemoAcceptanceTests(unittest.TestCase):
    def test_full_demo_in_fresh_project(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "speaker_diag",
                    "demo",
                    "--root",
                    str(root),
                    "--samples",
                    str(root / "samples"),
                ],
                cwd=str(project),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + "\n" + result.stdout)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "ok")
            self.assertTrue(all(summary["checks"].values()))
            self.assertEqual(
                set(summary["anomaly_codes"]),
                {"E_SPEC", "E_FIELD", "E_REF", "E_PORT", "E_HANDSHAKE", "E_LEASE", "E_REPLAY"},
            )
            report = json.loads((root / "artifacts" / "verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["counts"]["ledger_batches"], report["counts"]["batches"])


if __name__ == "__main__":
    unittest.main()

