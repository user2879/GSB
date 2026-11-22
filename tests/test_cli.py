from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_validate_command_returns_utf8_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "tabular_lab", "validate", "--config", "configs/experiment.yaml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["models"]), 3)
        self.assertGreater(payload["data"]["rows"], 0)

    def test_convert_command_roundtrip_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1_path = root / "models.v1.txt"
            sidecars = root / "sidecars"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tabular_lab",
                    "convert",
                    "--input",
                    "samples/models.v2.json",
                    "--to",
                    "v1",
                    "--output",
                    str(v1_path),
                    "--sidecar-dir",
                    str(sidecars),
                    "--tolerance",
                    "0.001",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["roundtrip"]["passed"])
            self.assertTrue(v1_path.exists())
            self.assertEqual(len(list(sidecars.glob("*.json"))), 3)

            v2_path = root / "roundtrip.v2.json"
            completed_back = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tabular_lab",
                    "convert",
                    "--input",
                    str(v1_path),
                    "--to",
                    "v2",
                    "--output",
                    str(v2_path),
                    "--sidecar-input",
                    str(sidecars),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(completed_back.returncode, 0, completed_back.stderr)
            models = json.loads(v2_path.read_text(encoding="utf-8"))["models"]
            self.assertEqual([model["name"] for model in models], ["sample_ridge", "sample_lasso", "sample_elastic"])
            self.assertEqual(models[2]["mix"], 0.55)

    def test_inspect_business_error_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.v1.txt"
            path.write_text("too-short\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "tabular_lab", "inspect", "--input", str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(completed.returncode, 2)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["error"]["code"], "E_FIELD_TRUNCATED")


if __name__ == "__main__":
    unittest.main()
