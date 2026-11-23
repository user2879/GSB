"""业务要求 7/8：交互预算语义、并发一致性与快照恢复判定。"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from retbench.concurrency import BoundedPool, Task, stable_task_key
from retbench.config import Command, load_config
from retbench.engine import Engine
from retbench.errors import E_INTERACT_BUDGET_EXCEEDED, E_SNAPSHOT_STALE
from retbench.fingerprint import digest_obj
from retbench.generator import generate_workspace


class InteractionSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=8, inject_count=1)

    def _engine(self, snap: str, budget: int = 200, work_ms: int = 0,
                workers: int = 2) -> Engine:
        config = load_config(os.path.join(self.temp, "config.yaml"))
        config.latency_budget_ms = budget
        config.work_ms_per_task = work_ms
        config.thread_pool_size = workers
        engine = Engine(config, os.path.join(self.temp, "samples"),
                        os.path.join(self.temp, snap))
        engine.load_samples()
        return engine, config

    def test_budget_exceeded_returns_stale_summary(self) -> None:
        engine, config = self._engine("snap_budget", budget=5, work_ms=20)
        pool = BoundedPool(config.thread_pool_size, config.work_ms_per_task)
        engine.baseline_report(pool)
        outcome = engine.handle_command(
            Command(seq=0, op="set_weight",
                    payload={"channel": "color", "value": 0.9}),
            1, pool,
        )
        self.assertTrue(outcome.budget_exceeded)
        self.assertTrue(outcome.stale)
        self.assertEqual(outcome.report["budget"]["code"],
                         E_INTERACT_BUDGET_EXCEEDED)
        self.assertIn("completed_samples", outcome.report["summary"])
        self.assertIn("channel_candidate_counts", outcome.report["summary"])

    def test_peak_concurrency_never_exceeds_limit(self) -> None:
        engine, config = self._engine("snap_pool", workers=3)
        pool = BoundedPool(3)
        engine.baseline_report(pool)
        self.assertLessEqual(pool.peak_concurrency, 3)
        self.assertGreaterEqual(pool.peak_concurrency, 1)

    def test_stable_sort_keys_are_deterministic(self) -> None:
        keys = [
            stable_task_key("s002", "recall"),
            stable_task_key("s001", "visual", (1, 0)),
            stable_task_key("s001", "visual", (0, 1)),
        ]
        self.assertEqual(sorted(keys), [
            ("s001", 0, 1, "visual"),
            ("s001", 1, 0, "visual"),
            ("s002", -1, -1, "recall"),
        ])

    def test_thread_pool_preserves_order(self) -> None:
        pool = BoundedPool(4)
        tasks = [Task(stable_task_key(f"s{i:02d}", "visual"),
                      lambda value=i: value * 2, f"task-{i}") for i in range(20)]
        results = pool.map_ordered(tasks)
        self.assertEqual([value for _, value in results], [i * 2 for i in range(20)])

    def test_snapshot_stale_detection_and_rebuild(self) -> None:
        snap = os.path.join(self.temp, "snap_resume")
        engine, config = self._engine(snap)
        pool = BoundedPool(2)
        engine.baseline_report(pool)
        manifest_path = os.path.join(snap, "manifest.json")
        self.assertTrue(os.path.isfile(manifest_path))

        same = Engine(config, os.path.join(self.temp, "samples"), snap)
        code, _ = same.resume_or_rebuild()
        self.assertIsNone(code)

        drifted_config = load_config(os.path.join(self.temp, "config.yaml"))
        drifted_config.weights = dict(config.weights)
        drifted_config.weights["edge"] = 0.75
        drifted = Engine(drifted_config, os.path.join(self.temp, "samples"), snap)
        stale_code, _ = drifted.resume_or_rebuild()
        self.assertEqual(stale_code, E_SNAPSHOT_STALE)
        self.assertFalse(os.path.isfile(manifest_path))

    def test_report_contains_required_sections(self) -> None:
        engine, _ = self._engine("snap_report")
        pool = BoundedPool(2)
        report = engine.baseline_report(pool)
        for section in ("per_query", "boundary_samples", "failures", "decision",
                        "ledger", "quantization_audit", "concurrency", "budget"):
            self.assertIn(section, report)
        self.assertTrue(report["failures"])


if __name__ == "__main__":
    unittest.main()
