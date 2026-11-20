import time
import unittest

from shardbench.config import load_config
from shardbench.errors import (
    OUTCOME_BUDGET_TIMEOUT,
    OUTCOME_CONFIG_MISSING,
    OUTCOME_DEPENDENCY_NOT_READY,
    OUTCOME_SUCCESS,
)
from shardbench.scheduler import ProbeTask, Scheduler

from .test_validation import sample_paths


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(*sample_paths())

    def _tasks(self, count=12):
        tasks = []
        for index in range(count):
            db = f"db{index % self.config.tables['t_user'].db_count}"
            tasks.append(
                ProbeTask(
                    task_id=f"task-{index:03d}",
                    unit_id=f"unit-{index:03d}",
                    dbs=(db,),
                    ready=True,
                    config_present=True,
                    order_index=index,
                )
            )
        return tasks

    def test_never_exceeds_global_and_per_db_budget(self):
        scheduler = Scheduler(self.config)
        traces = scheduler.run(self._tasks())
        verification = scheduler.verify_never_exceeded()
        self.assertTrue(verification["global_ok"])
        self.assertTrue(verification["per_db_ok"])
        self.assertLessEqual(
            verification["peak_global_running"],
            self.config.budget.global_concurrency,
        )
        for peak in verification["peak_per_db_running"].values():
            self.assertLessEqual(peak, self.config.budget.per_db_quota)
        self.assertTrue(all(trace.outcome_code == OUTCOME_SUCCESS for trace in traces))

    def test_wait_timing_recorded_with_monotonic_clock(self):
        scheduler = Scheduler(self.config)
        traces = scheduler.run(self._tasks())
        for trace in traces:
            self.assertIsNotNone(trace.wait_end_ms)
            self.assertIsNotNone(trace.run_start_ms)
            self.assertIsNotNone(trace.run_end_ms)
            self.assertGreaterEqual(trace.wait_end_ms, trace.wait_start_ms)
            self.assertGreaterEqual(trace.run_end_ms, trace.run_start_ms)

    def test_dependency_not_ready_outcome(self):
        tasks = self._tasks(3)
        tasks[1].ready = False
        scheduler = Scheduler(self.config)
        traces = scheduler.run(tasks)
        codes = {trace.unit_id: trace.outcome_code for trace in traces}
        self.assertEqual(codes[tasks[1].unit_id], OUTCOME_DEPENDENCY_NOT_READY)

    def test_config_missing_outcome(self):
        tasks = self._tasks(3)
        tasks[0].config_present = False
        scheduler = Scheduler(self.config)
        traces = scheduler.run(tasks)
        codes = {trace.unit_id: trace.outcome_code for trace in traces}
        self.assertEqual(codes[tasks[0].unit_id], OUTCOME_CONFIG_MISSING)

    def test_budget_timeout_outcome(self):
        config = load_config(*sample_paths())
        config.budget = type(config.budget)(
            global_concurrency=4,
            per_db_quota=1,
            wait_timeout_ms=5,
        )

        def slow_probe(task):
            time.sleep(0.05)
            return "done"

        tasks = self._tasks(12)
        # 全部压到同一物理库，每库配额 1 使其余任务必然排队并触发超时。
        for task in tasks:
            task.dbs = ("db0",)
        scheduler = Scheduler(config, probe_fn=slow_probe)
        traces = scheduler.run(tasks)
        codes = [trace.outcome_code for trace in traces]
        self.assertIn(OUTCOME_BUDGET_TIMEOUT, codes)

    def test_concurrent_interleaving_does_not_corrupt_counters(self):
        config = load_config(*sample_paths())
        config.budget = type(config.budget)(
            global_concurrency=4, per_db_quota=2, wait_timeout_ms=4000
        )
        scheduler = Scheduler(config)
        traces = scheduler.run(self._tasks(40))
        verification = scheduler.verify_never_exceeded()
        self.assertTrue(verification["global_ok"])
        self.assertTrue(verification["per_db_ok"])
        self.assertEqual(len(traces), 40)


if __name__ == "__main__":
    unittest.main()
