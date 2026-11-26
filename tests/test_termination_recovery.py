import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class TerminationRecoveryTests(unittest.TestCase):
    def test_sigterm_flush_and_snapshot_recovery(self):
        work = Path(tempfile.mkdtemp())
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + \
            env.get("PYTHONPATH", "")
        child = subprocess.Popen(
            [sys.executable, "-m", "tests.fxchild.term_child", str(work)],
            cwd=Path.cwd(), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 30
            line = ""
            while time.time() < deadline:
                line = child.stdout.readline().strip()
                if line == "ready":
                    break
            self.assertEqual(line, "ready")
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=15)
        finally:
            if child.poll() is None:
                child.kill()

        snap_path = work / "snap" / "snapshot.json"
        diag_path = work / "snap" / "diagnostics.json"
        self.assertTrue(snap_path.exists())
        self.assertTrue(diag_path.exists())
        snapshot = json.loads(snap_path.read_text())
        diagnostics = json.loads(diag_path.read_text())
        self.assertEqual(snapshot["global_version"], 1)
        self.assertEqual(sorted(snapshot["completed_frames"]), [0, 1])
        self.assertEqual(diagnostics["signal"], int(signal.SIGTERM))
        self.assertTrue(diagnostics["flushed"])


if __name__ == "__main__":
    unittest.main()
