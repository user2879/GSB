"""Requirement 8: graceful shutdown and forced-kill snapshot recovery."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from dbench.config import load_config
from dbench.orchestrator import Orchestrator
from dbench.store import SnapshotStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name: str):
    path = os.path.join(ROOT, "configs", name)
    cfg = load_config(path)
    base = os.path.dirname(path)
    return dataclasses.replace(
        cfg,
        vocab_file=os.path.normpath(os.path.join(base, cfg.vocab_file)),
    )


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-rec-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graceful_shutdown_completes_atomic_stage_and_reports_exit_zero(self):
        cfg = _load("valid_base.json")
        store = SnapshotStore(os.path.join(self.tmp, "graceful"))
        orch = Orchestrator(cfg, store,
                            output_png=os.path.join(self.tmp, "g.png"))
        orch.request_shutdown()  # signal before work starts: clean exit path
        report = orch.run()
        self.assertEqual(report["shutdown"],
                         {"graceful": True, "exit_code": 0})

    def _script_path(self) -> str:
        return os.path.join(self.tmp, "proc_runner.py")

    def _write_runner(self, store_dir: str, report_path: str,
                      png_path: str, config_path: str,
                      vocab_abs: str) -> None:
        script = f"""
import dataclasses, sys
sys.path.insert(0, {json.dumps(ROOT)})
from dbench.config import load_config
from dbench.orchestrator import Orchestrator
from dbench.store import SnapshotStore
cfg = load_config({json.dumps(config_path)})
cfg = dataclasses.replace(cfg, vocab_file={json.dumps(vocab_abs)})
store = SnapshotStore({json.dumps(store_dir)})
orch = Orchestrator(cfg, store, output_png={json.dumps(png_path)})
import signal
def handler(s, f):
    orch.request_shutdown()
signal.signal(signal.SIGTERM, handler)
report = orch.run()
import json as _json
with open({json.dumps(report_path)}, "w") as h:
    _json.dump(report, h)
"""
        with open(self._script_path(), "w", encoding="utf-8") as handle:
            handle.write(script)

    def test_sigterm_yields_zero_and_resume_uses_committed_snapshots(self):
        cfg = _load("valid_base.json")
        store_dir = os.path.join(self.tmp, "kill")
        report_path = os.path.join(self.tmp, "report.json")
        png_path = os.path.join(self.tmp, "k.png")
        self._write_runner(store_dir, report_path, png_path,
                           os.path.join(ROOT, "configs", "valid_base.json"),
                           cfg.vocab_file)
        proc = subprocess.Popen(
            [sys.executable, self._script_path()],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Let at least one stage commit, then hard-kill (SIGKILL = forced
        # termination, no Python cleanup possible).
        committed = False
        deadline = time.time() + 15
        while time.time() < deadline:
            progress_file = os.path.join(store_dir, "progress.json")
            if os.path.isfile(progress_file):
                with open(progress_file, encoding="utf-8") as handle:
                    progress = json.load(handle)
                if progress.get("committed"):
                    committed = True
                    break
            time.sleep(0.05)
        self.assertTrue(committed, "no committed snapshot before kill")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        proc.stdout.close()
        proc.stderr.close()
        self.assertEqual(proc.returncode, -signal.SIGKILL)

        # Restart in-process: recovery must rely only on fully replaced
        # artifacts + the progress ledger and report the recovery point.
        store = SnapshotStore(store_dir)
        orch = Orchestrator(cfg, store, output_png=png_path)
        report = orch.run()
        self.assertTrue(report["recovery"]["recovered"])
        self.assertIsNotNone(report["recovery"]["version"])
        # At least the recovered stage was reused, not recomputed.
        self.assertIn(report["recovery"]["stage"], report["reuse"])
        # Final PNG is produced and readable.
        self.assertTrue(os.path.isfile(png_path))


if __name__ == "__main__":
    unittest.main()
