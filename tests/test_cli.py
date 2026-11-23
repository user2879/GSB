from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT, SRC, make_sample


class CliTests(unittest.TestCase):
    def test_run_stop_recover_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            samples = base / "samples"
            make_sample(samples, budget=3)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SRC)
            command = [
                sys.executable,
                "-m",
                "inpaint_pyramid",
                "run",
                "--config",
                str(samples / "config.json"),
                "--commands",
                str(samples / "commands.json"),
                "--images",
                str(samples),
                "--state-dir",
                str(base / "state"),
                "--output-dir",
                str(base / "out1"),
                "--report",
                str(base / "report1.json"),
                "--stop-after",
                "2",
            ]
            first = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            recovered = subprocess.run(
                [sys.executable, "-m", "inpaint_pyramid", "recover", "--state-dir", str(base / "state")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
            recovery_info = json.loads(recovered.stdout)
            self.assertGreaterEqual(recovery_info["recovered_version"], 1)
            resume = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inpaint_pyramid",
                    "run",
                    "--resume",
                    "--config",
                    str(samples / "config.json"),
                    "--commands",
                    str(samples / "commands.json"),
                    "--images",
                    str(samples),
                    "--state-dir",
                    str(base / "state"),
                    "--output-dir",
                    str(base / "out2"),
                    "--report",
                    str(base / "report2.json"),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(resume.returncode, 0, resume.stderr + resume.stdout)
            resumed = json.loads(resume.stdout)
            self.assertEqual(resumed["status"], "OK")
            verify = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inpaint_pyramid",
                    "verify",
                    "--config",
                    str(samples / "config.json"),
                    "--commands",
                    str(samples / "commands.json"),
                    "--images",
                    str(samples),
                    "--report",
                    str(base / "verify.json"),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr + verify.stdout)
            summary = json.loads(verify.stdout)
            self.assertEqual(summary["status"], "ok")
            self.assertTrue(summary["final_comparison"]["equal"])


if __name__ == "__main__":
    unittest.main()

