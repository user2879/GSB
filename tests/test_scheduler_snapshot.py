import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

from texmat_pipeline.errors import E_POINT_BUDGET_PENDING, E_SNAPSHOT_STALE
from texmat_pipeline.models import load_generation_config, load_materials
from texmat_pipeline.pipeline import TexturePipeline
from texmat_pipeline.scheduler import Task, run_scheduler, run_serial_reference
from texmat_pipeline.snapshot import SnapshotStore
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_samples.py"


class SchedulerTests(unittest.TestCase):
    def _tasks(self):
        return [
            Task(layer_index=2, batch_index=0, point_count=100, payload="L2"),
            Task(layer_index=0, batch_index=1, point_count=30, payload="L0B1"),
            Task(layer_index=0, batch_index=0, point_count=50, payload="L0B0"),
            Task(layer_index=1, batch_index=0, point_count=40, payload="L1"),
        ]

    @staticmethod
    def _fn(task: Task) -> dict:
        return {"key": task.sort_key, "points": task.point_count}

    def test_stable_sort_key_and_layer_order(self):
        result = run_scheduler(self._tasks(), self._fn, threads=1, point_budget=0)
        keys = [t.sort_key for t, _ in result.results]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(keys, ["L000000-B000000", "L000000-B000001",
                                "L000001-B000000", "L000002-B000000"])

    def test_budget_exhaustion_marks_pending(self):
        # A single refill round: first task consumes the whole batch budget,
        # the remaining tasks stay pending in stable layer/batch order.
        tasks = [Task(0, 0, 60, "a"), Task(0, 1, 30, "b"),
                 Task(1, 0, 40, "c"), Task(2, 0, 100, "d")]
        result = run_scheduler(tasks, self._fn, threads=2, point_budget=60, max_refills=1)
        self.assertTrue(result.exhausted)
        pending_keys = [t.sort_key for t in result.pending]
        self.assertEqual(pending_keys, ["L000000-B000001", "L000001-B000000", "L000002-B000000"])
        self.assertTrue(all(f.code == E_POINT_BUDGET_PENDING for f in result.failures))
        self.assertEqual(result.budget_log, [("L000000-B000000", 60, 0)])

    def test_concurrent_matches_serial_budget_and_outputs(self):
        tasks = self._tasks()
        concurrent = run_scheduler(list(tasks), self._fn, threads=4, point_budget=80)
        serial = run_serial_reference(list(tasks), self._fn, point_budget=80)
        self.assertEqual(concurrent.budget_log, serial.budget_log)
        self.assertEqual([t.sort_key for t in concurrent.pending],
                         [t.sort_key for t in serial.pending])
        self.assertEqual([f.to_dict() for f in concurrent.failures],
                         [f.to_dict() for f in serial.failures])
        cc = {k: v for k, v in [(t.sort_key, o) for t, o in concurrent.results]}
        ss = {k: v for k, v in [(t.sort_key, o) for t, o in serial.results]}
        self.assertEqual(cc, ss)


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="texmat-snap-"))
        cls.sample = cls.tmp / "sample"
        subprocess.run(
            [sys.executable, str(GENERATOR), "--out", str(cls.sample),
             "--tiers", "16", "--layers", "3", "--point-budget", "1024"],
            check=True, capture_output=True,
        )
        cls.config, _ = load_generation_config(cls.sample / "generation_config.json")
        cls.materials, _ = load_materials(cls.sample / "materials.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _pipeline(self, name: str, config=None) -> TexturePipeline:
        work = self.tmp / name
        if work.exists():
            shutil.rmtree(work)
        return TexturePipeline(self.sample, work, config or self.config, self.materials)

    def test_snapshot_commit_resume_and_stale_rebuild(self):
        pipe = self._pipeline("snap")
        result = pipe.run(16)
        store = pipe.snapshots
        state = store.inspect(self.config.config_hash, self.config.fixed_point.frac_bits)
        self.assertEqual(state.global_version, 1)
        self.assertFalse(state.stale)
        self.assertEqual(state.completed_batches, len(self.config.layers))
        current = store.current_dir()
        self.assertTrue((current / "manifest.json").exists())
        self.assertTrue((current / "report.json").exists())

        # Config hash change must be detected as stale and trigger a rebuild.
        changed = self.config.with_overrides(blend_mode="weighted_sum")
        stale_state = store.inspect(changed.config_hash, changed.fixed_point.frac_bits)
        self.assertTrue(stale_state.stale)
        self.assertEqual(stale_state.failure.code, E_SNAPSHOT_STALE)

        pipe2 = TexturePipeline(self.sample, self.tmp / "snap", changed, self.materials)
        rebuilt = pipe2.run(16)
        state2 = pipe2.snapshots.inspect(changed.config_hash, changed.fixed_point.frac_bits)
        self.assertFalse(state2.stale)
        # rebuild created a new version rather than overwriting files
        self.assertEqual(state2.global_version, 2)
        report = json.loads((Path(rebuilt.snapshots[0]) / "report.json").read_text())
        self.assertEqual(report["blend_mode"], "weighted_sum")

    def test_snapshot_files_are_atomic_versions(self):
        pipe = self._pipeline("atomic")
        pipe.run(16)
        versions = [p for p in (self.tmp / "atomic" / "snapshots" / "versions").iterdir() if p.is_dir()]
        self.assertEqual(len(versions), 1)
        manifest = versions[0] / "manifest.json"
        self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
