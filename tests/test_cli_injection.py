from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from alert_pipeline.samplegen import generate_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_temporary = tempfile.TemporaryDirectory()
        cls.class_root = Path(cls.class_temporary.name)
        cls.samples = cls.class_root / "samples"
        generate_samples(cls.samples, seed=20261122)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.class_temporary.cleanup()

    def test_three_injection_business_codes_and_structured_logs(self) -> None:
        cases = {
            "corrupt": ("E_SPEC", 2, 41, "$"),
            "unsupported-version": ("E_FIELD", 3, 1, "version"),
            "window-conflict": ("E_WINDOW", 4, 1, "silence_window_ms"),
        }
        for mode, (code, exit_code, line, field) in cases.items():
            with self.subTest(mode=mode):
                database = self.class_root / f"{mode}.db"
                log = self.class_root / f"{mode}.jsonl"
                output = self.class_root / f"{mode}-out"
                command = [
                    sys.executable,
                    "-m",
                    "alert_pipeline",
                    "run",
                    "--manifest",
                    str(self.samples / "manifest.json"),
                    "--inject",
                    mode,
                    "--database",
                    str(database),
                    "--log",
                    str(log),
                    "--output-dir",
                    str(output),
                ]
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, exit_code, completed.stderr)
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["business_code"], code)
                self.assertEqual(payload["line"], line)
                self.assertEqual(payload["field"], field)
                log_rows = [
                    json.loads(line_value)
                    for line_value in log.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(log_rows[-1]["business_code"], code)
                self.assertEqual(log_rows[-1]["event"], "error")
                self.assertEqual(log_rows[-1]["inject"], mode)


if __name__ == "__main__":
    unittest.main()
