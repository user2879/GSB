from __future__ import annotations

import time
import unittest

from mvcc_workbench.scheduler import ScheduledTask, run_scheduled_tasks


class SchedulerTests(unittest.TestCase):
    def test_dual_quota_and_real_threads(self) -> None:
        tasks = [
            ScheduledTask(
                task_id=f"T{index:02d}",
                read_point_id=f"RP{index // 4}",
                fn=lambda: time.sleep(0.02) or "ok",
            )
            for index in range(16)
        ]
        report = run_scheduled_tasks(
            tasks,
            global_concurrency=3,
            per_read_point_concurrency=2,
        )
        self.assertTrue(report.budget["constraints_ok"])
        self.assertLessEqual(report.observed_peaks["global"], 3)
        self.assertEqual(report.observed_peaks["global"], 3)
        self.assertTrue(all(value <= 2 for value in report.observed_peaks["per_read_point"].values()))
        self.assertIn(2, report.observed_peaks["per_read_point"].values())
        self.assertGreaterEqual(report.observed_peaks["global"], 2)
        self.assertGreater(report.totals["thread_count"], 1)
        self.assertGreater(report.totals["blocked_task_count"], 0)
        for trace in report.traces:
            self.assertIsNotNone(trace["started_at_ms"])
            self.assertIsNotNone(trace["finished_at_ms"])
            self.assertGreaterEqual(trace["finished_at_ms"], trace["started_at_ms"])
            if trace["quota_wait_start_ms"] is not None:
                self.assertGreaterEqual(
                    trace["quota_wait_end_ms"], trace["quota_wait_start_ms"]
                )

    def test_results_keep_task_identity(self) -> None:
        tasks = [
            ScheduledTask(task_id="B", read_point_id="R", fn=lambda: 2),
            ScheduledTask(task_id="A", read_point_id="R", fn=lambda: 1),
        ]
        report = run_scheduled_tasks(
            tasks, global_concurrency=2, per_read_point_concurrency=1
        )
        self.assertEqual(report.results, {"A": 1, "B": 2})


if __name__ == "__main__":
    unittest.main()
