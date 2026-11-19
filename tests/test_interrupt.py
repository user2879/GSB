from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import ROOT, description, event, write_description


class InterruptRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _command(self, path: Path, state: Path, output: Path, delay: int) -> list[str]:
        return [
            sys.executable,
            "-m",
            "subtitle_engine",
            "--input",
            str(path),
            "--state-dir",
            str(state),
            "--output-dir",
            str(output),
            "--block-delay-ms",
            str(delay),
        ]

    def _wait_for_block(self, proc: subprocess.Popen[str], state: Path) -> list[Path]:
        deadline = time.time() + 15
        while time.time() < deadline:
            files = sorted(state.glob("*/blocks/[0-9]*.npz"))
            if files:
                return files
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                self.fail(f"engine exited early: {proc.returncode}\n{stdout}\n{stderr}")
            time.sleep(0.02)
        self.fail("timed out waiting for the first persisted block")
        return []

    def _description(self, name: str) -> dict:
        return description(
            name=name,
            events=[
                event("e1", 0, 600, "First subtitle line"),
                event("e2", 600, 1200, "Second subtitle line", anchor="top_left"),
            ],
            duration_ms=1200,
            concurrency=1,
        )

    def test_sigterm_exits_at_boundary_and_resumes(self) -> None:
        value = self._description("graceful")
        path = write_description(self.root, "graceful", value)
        state = self.root / "graceful" / "state"
        output = self.root / "graceful" / "output"
        proc = subprocess.Popen(
            self._command(path, state, output, 120),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        completed = self._wait_for_block(proc, state)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=20)
        self.assertEqual(proc.returncode, 130, stdout + stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "interrupted")
        self.assertTrue(list(state.glob("*/interrupted.json")))
        self.assertFalse((output / "frames.npy").exists())
        persisted = len(list(state.glob("*/blocks/[0-9]*.npz")))
        self.assertGreaterEqual(persisted, len(completed))

        resumed = subprocess.run(
            self._command(path, state, output, 0),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        report = json.loads((output / "report.json").read_text())
        self.assertGreaterEqual(report["state"]["reused_block_count"], 1)
        self.assertEqual(report["frame_count"], 60)
        self.assertFalse(list(state.glob("*/interrupted.json")))

    @unittest.skipIf(os.name == "nt", "forceful POSIX signal test")
    def test_force_kill_reuses_only_valid_persisted_blocks(self) -> None:
        value = self._description("forced")
        path = write_description(self.root, "forced", value)
        state = self.root / "forced" / "state"
        output = self.root / "forced" / "output"
        proc = subprocess.Popen(
            self._command(path, state, output, 120),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_for_block(proc, state)
        proc.kill()
        proc.communicate(timeout=20)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((output / "frames.npy").exists())

        resumed = subprocess.run(
            self._command(path, state, output, 0),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        report = json.loads((output / "report.json").read_text())
        self.assertGreaterEqual(report["state"]["reused_block_count"], 1)
        self.assertEqual(report["state"]["reused_block_count"] + report["state"]["recomputed_block_count"], report["block_count"])


if __name__ == "__main__":
    unittest.main()
