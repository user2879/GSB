from __future__ import annotations

import threading
import time
import unittest

from mvcc_workbench.budget import Allocation
from mvcc_workbench.model import Budget
from mvcc_workbench.scheduler import QuotaScheduler
from mvcc_workbench.visibility import compute_visibility

from tests.helpers import base_dataset


class SchedulerTests(unittest.TestCase):
    def test_global_and_per_read_point_quota_are_enforced(self) -> None:
        dataset = base_dataset()
        allocations = (
            Allocation("Q01", "RP01", "acct:1", 10, 4),
            Allocation("Q02", "RP01", "acct:2", 20, 7),
            Allocation("Q03", "RP02", "acct:2", 10, 5),
        )
        budget = Budget(
            global_cost_budget=100,
            round_cost_budget=100,
            global_concurrency=2,
            per_read_point_concurrency=1,
            task_timeout_ms=2000,
        )
        lock = threading.Lock()
        active = 0
        active_by_rp: dict[str, int] = {}
        observed_global = 0
        observed_rp: dict[str, int] = {}

        def operation(allocation: Allocation, deadline_ns: int):
            nonlocal active, observed_global
            with lock:
                active += 1
                active_by_rp[allocation.read_point_id] = active_by_rp.get(allocation.read_point_id, 0) + 1
                observed_global = max(observed_global, active)
                observed_rp[allocation.read_point_id] = max(
                    observed_rp.get(allocation.read_point_id, 0),
                    active_by_rp[allocation.read_point_id],
                )
            try:
                time.sleep(0.02)
                return compute_visibility(dataset, dataset.queries_by_id[allocation.query_id], deadline_ns)
            finally:
                with lock:
                    active -= 1
                    active_by_rp[allocation.read_point_id] -= 1

        scheduler = QuotaScheduler(budget)
        traces = scheduler.run(allocations, operation)
        self.assertEqual(len(traces), 3)
        self.assertTrue(all(trace.status == "OK" for trace in traces))
        self.assertLessEqual(observed_global, 2)
        self.assertLessEqual(observed_rp["RP01"], 1)
        self.assertLessEqual(scheduler.max_observed_global, 2)
        self.assertLessEqual(scheduler.max_observed_per_read_point["RP01"], 1)
        self.assertTrue(any(trace.wait_ms >= 0 for trace in traces))

    def test_real_timeout_marks_task_without_a_conclusion(self) -> None:
        dataset = base_dataset()
        allocation = Allocation("Q01", "RP01", "acct:1", 10, 4)
        budget = Budget(
            global_cost_budget=100,
            round_cost_budget=100,
            global_concurrency=1,
            per_read_point_concurrency=1,
            task_timeout_ms=5,
        )

        def slow_operation(allocation: Allocation, deadline_ns: int):
            time.sleep(0.03)
            return compute_visibility(dataset, dataset.queries_by_id[allocation.query_id], deadline_ns)

        trace = QuotaScheduler(budget).run((allocation,), slow_operation)[0]
        self.assertEqual(trace.status, "TIMEOUT")
        self.assertIsNone(trace.decision)
        self.assertGreater(trace.duration_ms, trace.timeout_ms)


if __name__ == "__main__":
    unittest.main()
