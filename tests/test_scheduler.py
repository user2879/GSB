"""Requirement 7: global and per-worker concurrency quotas."""
from __future__ import annotations

import os
import threading
import unittest

from grayroute.batching import plan_batches
from grayroute.scheduler import Scheduler
from grayroute.sharding import attribute_all
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(__file__))


class SchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(os.path.join(ROOT, "config"),
                             os.path.join(ROOT, "data", "requests.ndjson"))
        cls.batches = plan_batches(cls.spec, attribute_all(cls.spec))

    def test_quotas_never_exceeded(self):
        lock = threading.Lock()
        in_flight = {"global": 0}
        per_worker: dict[str, int] = {}
        max_seen = {"global": 0}
        max_worker: dict[str, int] = {}

        def worker(batch):
            with lock:
                in_flight["global"] += 1
                per_worker[batch.worker_id] = \
                    per_worker.get(batch.worker_id, 0) + 1
                max_seen["global"] = max(max_seen["global"],
                                         in_flight["global"])
                max_worker[batch.worker_id] = max(
                    max_worker.get(batch.worker_id, 0),
                    per_worker[batch.worker_id])
            threading.Event().wait(0.01)
            with lock:
                in_flight["global"] -= 1
                per_worker[batch.worker_id] -= 1

        scheduler = Scheduler(max_global=2, per_worker_quota=1)
        runs = scheduler.run(self.batches, worker)
        self.assertEqual(len(runs), len(self.batches))
        self.assertLessEqual(max_seen["global"], 2)
        for worker_id, peak in max_worker.items():
            self.assertLessEqual(peak, 1, worker_id)
        for run in runs:
            self.assertGreaterEqual(run.wait_end_ms, run.wait_start_ms)
            self.assertGreaterEqual(run.run_end_ms, run.run_start_ms)

    def test_all_batches_executed_once(self):
        seen = []
        lock = threading.Lock()

        def worker(batch):
            with lock:
                seen.append(batch.batch_id)

        Scheduler(3, 2).run(self.batches, worker)
        self.assertEqual(sorted(seen),
                         sorted(b.batch_id for b in self.batches))


if __name__ == "__main__":
    unittest.main()
