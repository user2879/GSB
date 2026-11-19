from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass

from bucketpack.errors import E_WIDTH, BusinessError
from bucketpack.scheduler import schedule_packs
from bucketpack.spec import PackingSpec


@dataclass(frozen=True)
class Job:
    pack_id: str
    source_id: str
    width_used: int


class SchedulerTests(unittest.TestCase):
    def test_global_budget_and_source_quota_hold(self) -> None:
        packing = PackingSpec(
            rows_per_pack=1,
            pack_width_budget=100,
            global_width_budget=90,
            pack_quota_per_source=1,
            workers=4,
        )
        jobs = [
            Job("a:p0", "a", 60),
            Job("a:p1", "a", 60),
            Job("b:p0", "b", 30),
        ]
        lock = threading.Lock()
        active_total = 0
        max_total = 0
        active_source = {"a": 0, "b": 0}
        max_source = {"a": 0, "b": 0}

        def compute(job: Job) -> str:
            nonlocal active_total, max_total
            with lock:
                active_total += job.width_used
                active_source[job.source_id] += 1
                max_total = max(max_total, active_total)
                max_source[job.source_id] = max(max_source[job.source_id], active_source[job.source_id])
            time.sleep(0.01)
            with lock:
                active_total -= job.width_used
                active_source[job.source_id] -= 1
            return job.pack_id

        result, values = schedule_packs(jobs, packing, compute)
        self.assertEqual(set(values), {"a:p0", "a:p1", "b:p0"})
        self.assertLessEqual(max_total, 90)
        self.assertLessEqual(max_source["a"], 1)
        self.assertLessEqual(max_source["b"], 1)
        self.assertLessEqual(result.max_active_width, 90)
        self.assertEqual([item.pack_id for item in result.entries], ["a:p0", "a:p1", "b:p0"])
        for entry in result.entries:
            self.assertLessEqual(entry.start_ms, entry.end_ms)
            self.assertGreaterEqual(entry.wait_ms, 0.0)

    def test_ready_scan_is_order_independent(self) -> None:
        packing = PackingSpec(1, 100, 90, 1, 2)
        jobs = [Job("a", "a", 80), Job("b", "b", 30), Job("c", "c", 20)]
        forward, _ = schedule_packs(jobs, packing, lambda job: job.pack_id)
        reverse, _ = schedule_packs(list(reversed(jobs)), packing, lambda job: job.pack_id)
        self.assertEqual(
            [(item.pack_id, item.width_used) for item in forward.entries],
            [(item.pack_id, item.width_used) for item in reverse.entries],
        )
        self.assertEqual(forward.max_active_width, reverse.max_active_width)

    def test_job_larger_than_global_budget_rejected(self) -> None:
        packing = PackingSpec(1, 100, 50, 1, 1)
        with self.assertRaises(BusinessError) as caught:
            schedule_packs([Job("a", "a", 60)], packing, lambda job: job.pack_id)
        self.assertEqual(caught.exception.code, E_WIDTH)


if __name__ == "__main__":
    unittest.main()
