from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from tests.helpers import PROJECT_ROOT, create_sample

import repair_studio  # noqa: F401
from repair_studio.engine import recover_output
from repair_studio.rendering import Renderer
from repair_studio.snapshot import SnapshotStore


class SnapshotAndExitTests(unittest.TestCase):
    def test_snapshot_version_detection_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root / "samples", "snapshot", width=48, height=36)
            output = root / "output"
            renderer = Renderer(output / "cache")
            try:
                first_outcome = renderer.render(sample, sample.config.params)
                changed_params = replace(sample.config.params, quant_levels=32)
                second_outcome = renderer.render(sample, changed_params)
            finally:
                renderer.close()

            store = SnapshotStore(output)
            first = store.commit(
                sample.sample_id,
                sample.config.fingerprint(),
                1,
                first_outcome,
            )
            self.assertEqual(first["status"], "committed")
            self.assertEqual(first["global_version"], 1)
            same = store.commit(
                sample.sample_id,
                sample.config.fingerprint(),
                1,
                first_outcome,
            )
            self.assertEqual(same["status"], "unchanged")
            self.assertEqual(same["global_version"], 1)
            second = store.commit(
                sample.sample_id,
                sample.config.fingerprint(),
                2,
                second_outcome,
            )
            self.assertEqual(second["status"], "committed")
            self.assertEqual(second["global_version"], 2)
            self.assertTrue((store.sample_root(sample.sample_id) / "gen-00000001").is_dir())
            self.assertTrue((store.sample_root(sample.sample_id) / "gen-00000002").is_dir())

            incomplete = store.sample_root(sample.sample_id) / ".tmp-snapshot-killed"
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text('{"complete":false}', encoding="utf-8")
            recovered = recover_output(output)
            self.assertEqual(recovered["status"], "recovered")
            self.assertEqual(recovered["samples"][0]["recovered_version"], 2)
            self.assertTrue((output / "recovered" / sample.sample_id / "final.png").is_file())
            self.assertTrue((output / "recovered" / "recovery-report.json").is_file())

    def test_forced_kill_recovers_only_complete_snapshot_without_exit_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root / "samples", "killed", width=44, height=32)
            output = root / "output"
            script = f"""
import os, sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
import repair_studio
from pathlib import Path
from repair_studio.imaging import load_sample
from repair_studio.rendering import Renderer
from repair_studio.snapshot import SnapshotStore
sample = load_sample(Path({str(sample.directory)!r}))
renderer = Renderer(Path({str(output)!r}) / 'cache')
outcome = renderer.render(sample, sample.config.params)
SnapshotStore(Path({str(output)!r})).commit(
    sample.sample_id, sample.config.fingerprint(), 1, outcome
)
os._exit(137)
"""
            process = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 137)
            self.assertFalse((output / "reports").exists())
            recovered = recover_output(output)
            self.assertEqual(recovered["status"], "recovered")
            self.assertEqual(recovered["samples"][0]["completed_command_cursor"], 1)
            self.assertFalse(recovered["exit_report_written"])

    def test_sigterm_completes_snapshot_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            source = PROJECT_ROOT / "samples" / "procedural_a"
            shutil.copytree(source, samples / "slow")
            output = root / "output"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "repair_studio",
                    "run",
                    "--samples",
                    str(samples),
                    "--output",
                    str(output),
                    "--mode",
                    "sync",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.35)
            process.terminate()
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, msg=f"stdout={stdout}\nstderr={stderr}")
            reports = list((output / "reports").glob("diagnostics-*.json"))
            self.assertTrue(reports)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["failures"], [])
            recovered = recover_output(output)
            self.assertEqual(recovered["status"], "recovered")


if __name__ == "__main__":
    unittest.main()

