import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from vdu.synth import generate_demo


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vdu_recovery_")
        self.samples = os.path.join(self.root, "samples")
        self.configs = os.path.join(self.root, "configs")
        self.snapshots = os.path.join(self.root, "snapshots")
        generate_demo(self.samples, self.configs)
        config_path = os.path.join(self.configs, "calibration.json")
        with open(config_path) as handle:
            config = json.load(handle)
        # One segment per batch and a real delay so SIGKILL lands mid-run.
        config["scheduler"]["batch_budget"] = 4
        config["scheduler"]["test_step_delay"] = 0.4
        with open(config_path, "w") as handle:
            json.dump(config, handle)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _start(self):
        cmd = [sys.executable, "-m", "vdu", "run",
               "--samples", self.samples,
               "--config", os.path.join(self.configs, "calibration.json"),
               "--snapshots", self.snapshots]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_sigkill_then_recovery_uses_persisted_state(self):
        proc = self._start()
        deadline = time.time() + 15
        current = os.path.join(self.snapshots, 'current')
        while time.time() < deadline and not os.path.islink(current):
            time.sleep(0.1)
        self.assertTrue(os.path.islink(current))
        time.sleep(0.2)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        self.assertEqual(proc.returncode, -signal.SIGKILL)

        current = os.path.join(self.snapshots, "current")
        self.assertTrue(os.path.islink(current))
        partial_state = os.path.join(current, "state.json")
        self.assertTrue(os.path.exists(partial_state))
        with open(partial_state) as handle:
            state = json.load(handle)
        self.assertFalse(state["complete"])
        self.assertTrue(state["todo"])
        # tmp staging dir must be absent after atomic swap semantics.
        self.assertFalse(os.path.exists(os.path.join(self.snapshots, "tmp")))

        # Resume relies solely on the persisted snapshot, no in-memory state.
        resumed = self._start()
        out, err = resumed.communicate(timeout=60)
        self.assertEqual(resumed.returncode, 0, err.decode())
        payload = json.loads(out.decode())
        resumed.stdout.close()
        resumed.stderr.close()
        self.assertEqual(payload["state"]["todo"], [])
        self.assertTrue(payload["state"]["complete"])
        self.assertEqual(len(payload["segments"]), 4)

    def test_sigterm_writes_exit_report(self):
        proc = self._start()
        time.sleep(0.4)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        proc.stdout.close()
        proc.stderr.close()
        report = os.path.join(self.snapshots, "current", "exit_report.json")
        self.assertTrue(os.path.exists(report))
        with open(report) as handle:
            payload = json.load(handle)
        self.assertIn("reason", payload)
