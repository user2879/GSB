"""CLI JSON protocol, exit codes, interrupt/resume and signal handling."""

from __future__ import annotations

import json
import time
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_solver.py"


def valid_spec():
    return {
        "width": 12, "height": 10, "seed": 17,
        "blocked": [[0, 0], [11, 9]],
        "concurrency": 3, "memory_budget": 6, "max_rollback": 8,
        "alternative_limit": 24,
        "constraints": [
            {"id": "rooms", "type": "min_rooms",
             "params": {"count": 3, "min_room_w": 2, "min_room_h": 2,
                        "max_room_w": 3, "max_room_h": 3}},
            {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
            {"id": "joined", "type": "connectivity", "params": {}},
            {"id": "reach", "type": "reachability",
             "params": {"start": [1, 1], "end": [10, 8]}},
        ],
    }


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lvl-cli-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, payload):
        path = Path(self.tmp) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _run(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            capture_output=True, text=True, env=env, cwd=self.tmp,
        )

    def test_invalid_spec_returns_error_json_and_no_files(self):
        spec_path = self._write("bad.json", {"width": -1, "height": 4, "seed": 1,
                                             "constraints": []})
        snap_dir = str(Path(self.tmp) / "snaps")
        proc = self._run("solve", "--spec", spec_path, "--snapshot-dir", snap_dir)
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(proc.stderr)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["error"]["code"].startswith("E_"))
        self.assertFalse(Path(snap_dir).exists())

    def test_solve_writes_output_and_snapshots(self):
        spec_path = self._write("spec.json", valid_spec())
        snap_dir = str(Path(self.tmp) / "snaps")
        out_path = str(Path(self.tmp) / "result.json")
        proc = self._run("solve", "--spec", spec_path, "--snapshot-dir",
                         snap_dir, "--output", out_path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(Path(out_path).exists())
        self.assertTrue(list(Path(snap_dir, "current").glob("snapshot_*.json")))

    def test_interrupt_exit_code_then_resume_matches_full_run(self):
        spec_path = self._write("spec.json", valid_spec())
        full_dir = str(Path(self.tmp) / "full")
        full = self._run("solve", "--spec", spec_path, "--snapshot-dir", full_dir)
        self.assertEqual(full.returncode, 0)
        full_grid = json.loads(full.stdout)["grid"]

        split_dir = str(Path(self.tmp) / "split")
        interrupted = self._run(
            "solve", "--spec", spec_path, "--snapshot-dir", split_dir,
            "--interrupt-after", "2",
        )
        self.assertEqual(interrupted.returncode, 130)
        marker = Path(split_dir, "current", "interrupted.json")
        self.assertTrue(marker.exists())

        resumed = self._run("solve", "--spec", spec_path, "--snapshot-dir", split_dir)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        resumed_payload = json.loads(resumed.stdout)
        self.assertTrue(resumed_payload["resumed"])
        self.assertEqual(resumed_payload["grid"], full_grid)

    def test_verify_detects_tampering(self):
        spec_path = self._write("spec.json", valid_spec())
        snap_dir = str(Path(self.tmp) / "snaps")
        self._run("solve", "--spec", spec_path, "--snapshot-dir",
                  snap_dir, "--interrupt-after", "1")
        target = sorted(Path(snap_dir, "current").glob("snapshot_*.json"))[0]
        document = json.loads(target.read_text())
        document["digest"] = "deadbeef"
        target.write_text(json.dumps(document))
        proc = self._run("verify", "--spec", spec_path, "--snapshot-dir", snap_dir)
        self.assertEqual(proc.returncode, 3)
        self.assertFalse(json.loads(proc.stdout)["ok"])

    @unittest.skipIf(sys.platform.startswith("win"), "POSIX signal semantics only")
    def test_sigint_suspends_and_resumes(self):
        # A large alternative budget gives the child enough work to receive
        # the signal before finishing naturally.
        spec = valid_spec()
        spec["alternative_limit"] = 200
        spec["constraints"][0]["params"]["count"] = 6
        spec_path = self._write("big.json", spec)
        snap_dir = str(Path(self.tmp) / "signals")
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER), "solve", "--spec", spec_path,
             "--snapshot-dir", snap_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd=self.tmp,
        )
        time.sleep(0.3)
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 130, stderr)
        self.assertTrue(Path(snap_dir, "current", "interrupted.json").exists())
        resumed = self._run("solve", "--spec", spec_path, "--snapshot-dir", snap_dir)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertTrue(json.loads(resumed.stdout)["resumed"])


if __name__ == "__main__":
    unittest.main()
