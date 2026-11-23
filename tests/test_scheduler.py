from __future__ import annotations

import unittest

from drift_diag.canonical import canonical_json
from drift_diag.scheduler import schedule_decisions

from .helpers import complete_actual, config, decisions_for, target_v3


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.target = target_v3()
        self.actual = complete_actual(self.target)

    def test_global_and_per_workload_limits_with_real_workers(self):
        decisions, trace = schedule_decisions(
            self.target,
            self.actual,
            config(global_limit=6, per_workload_quota=2, task_delay_ms=3.0),
        )
        self.assertEqual(trace["global_active_peak"], 6)
        self.assertLessEqual(trace["global_active_peak"], 6)
        self.assertTrue(trace["global_limit_respected"])
        self.assertTrue(trace["quota_respected"])
        self.assertTrue(trace["assignments_all_on_workers"])
        self.assertGreaterEqual(trace["global_active_peak"], 2)
        for peak in trace["per_workload_active_peak"].values():
            self.assertLessEqual(peak, 2)
        assignments = [
            assignment
            for round_info in trace["rounds"]
            for assignment in round_info["assignments"]
        ]
        self.assertEqual(len(assignments), len(self.target.workloads) * 6)
        self.assertTrue(all(item["worker_ident"] != trace["main_thread_ident"] for item in assignments))
        self.assertEqual(canonical_json(decisions), canonical_json(decisions_for(self.target, self.actual)))

        quota_decisions, quota_trace = schedule_decisions(
            self.target,
            self.actual,
            config(global_limit=6, per_workload_quota=1, task_delay_ms=3.0),
        )
        quota_assignments = [
            assignment
            for round_info in quota_trace["rounds"]
            for assignment in round_info["assignments"]
        ]
        self.assertTrue(quota_trace["quota_respected"])
        self.assertLessEqual(quota_trace["global_active_peak"], 6)
        self.assertTrue(any(item["quota_wait_ms"] > 0 for item in quota_assignments))
        self.assertEqual(canonical_json(quota_decisions), canonical_json(decisions))

    def test_concurrency_layout_does_not_change_decision_set(self):
        first, _ = schedule_decisions(
            self.target,
            self.actual,
            config(global_limit=2, per_workload_quota=1, task_delay_ms=1.0),
        )
        second, _ = schedule_decisions(
            self.target,
            self.actual,
            config(global_limit=5, per_workload_quota=2, task_delay_ms=1.0),
        )
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_trace_has_rounds_waits_and_concurrency_counts(self):
        _, trace = schedule_decisions(
            self.target,
            self.actual,
            config(global_limit=4, per_workload_quota=1, task_delay_ms=2.0),
        )
        self.assertGreaterEqual(len(trace["rounds"]), 1)
        for round_info in trace["rounds"]:
            self.assertIn("assignments", round_info)
            self.assertIn("global_active_peak_so_far", round_info)
        for event in trace["concurrency_trajectory"]:
            self.assertLessEqual(event["global_active"], trace["global_limit"])
            self.assertLessEqual(event["workload_active"], 1)
        for round_info in trace["rounds"]:
            for assignment in round_info["assignments"]:
                self.assertLessEqual(assignment["queue_wait_start_ms"], assignment["queue_wait_end_ms"])
                self.assertLessEqual(assignment["quota_wait_start_ms"], assignment["quota_wait_end_ms"])


if __name__ == "__main__":
    unittest.main()
