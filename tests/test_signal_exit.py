import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SignalExitTest(unittest.TestCase):
    def test_sigterm_completes_snapshot_and_exits_zero(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        with tempfile.TemporaryDirectory() as tmp:
            env["REX_SIGNAL_TEST"] = tmp
            helper = Path(tmp) / "signal_helper.py"
            helper.write_text(
                "import os, time\n"
                "from pathlib import Path\n"
                "from rexapp.app import run_interactive\n"
                "def slow(d, s):\n"
                "    time.sleep(0.2)\n"
                "out = Path(os.environ['REX_SIGNAL_TEST'])\n"
                "run_interactive('samples', out, work_hook=slow)\n",
                encoding="utf-8",
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(helper),
                ],
                env=env,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.4)
            proc.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail("process did not exit after SIGTERM")
            self.assertEqual(proc.returncode, 0, stderr.decode())


if __name__ == "__main__":
    unittest.main()
