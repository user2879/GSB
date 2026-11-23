import unittest

from hotblock_diag.errors import (E_QBLOCK_BUDGET_EXHAUSTED,
                                  E_SCORE_BUDGET_EXHAUSTED, FailureList)
from hotblock_diag.scheduler import ScoreTask, Scheduler


class SchedulerTests(unittest.TestCase):
    def _tasks(self) -> list[ScoreTask]:
        return [ScoreTask(qb, kb, {}) for qb in range(3) for kb in range(4)]

    def test_global_budget_exhaustion_keeps_pending_queue(self):
        failures = FailureList()
        scheduler = Scheduler(threads=4, batch_budget=5, qblock_quota=10,
                              failures=failures)
        batch = scheduler.run_batch(self._tasks(), lambda task: task.key)
        self.assertEqual(len(batch.results), 5)
        self.assertTrue(batch.global_exhausted)
        self.assertEqual(len(batch.pending), 7)
        self.assertIn(E_SCORE_BUDGET_EXHAUSTED, failures.codes())
        keys = [(item["qb"], item["kb"]) for item in batch.deductions]
        self.assertEqual(keys, sorted(keys))

    def test_qblock_quota_triggers_dedicated_code(self):
        failures = FailureList()
        scheduler = Scheduler(threads=2, batch_budget=100, qblock_quota=2,
                              failures=failures)
        batch = scheduler.run_batch(self._tasks(), lambda task: task.key)
        self.assertEqual(batch.exhausted_q, {0, 1, 2})
        self.assertIn(E_QBLOCK_BUDGET_EXHAUSTED, failures.codes())
        self.assertEqual(len(batch.results), 6)

    def test_deduction_matches_serial_stable_order_under_threads(self):
        tasks = self._tasks()
        serial_failures = FailureList()
        serial = Scheduler(1, 100, 100, serial_failures)
        serial_batch = serial.run_batch(tasks, lambda task: task.key)

        parallel_failures = FailureList()
        parallel = Scheduler(8, 100, 100, parallel_failures)
        parallel_batch = parallel.run_batch(tasks, lambda task: task.key)
        self.assertEqual(
            [(d["qb"], d["kb"]) for d in serial_batch.deductions],
            [(d["qb"], d["kb"]) for d in parallel_batch.deductions],
        )

    def test_pending_tasks_priority_next_batch(self):
        failures = FailureList()
        scheduler = Scheduler(threads=2, batch_budget=4, qblock_quota=100,
                              failures=failures)
        first = scheduler.run_batch(self._tasks(), lambda task: task.key)
        self.assertEqual(len(first.pending), 8)
        second = scheduler.run_batch(first.pending, lambda task: task.key)
        self.assertEqual(len(second.results), 4)
        self.assertEqual(len(second.pending), 4)
        third = scheduler.run_batch(second.pending, lambda task: task.key)
        self.assertEqual(len(third.results), 4)
        self.assertEqual(scheduler.total_consumed, 12)
        full_seq = [(d["qb"], d["kb"]) for d in scheduler.total_deductions]
        self.assertEqual(full_seq, sorted(full_seq))


if __name__ == "__main__":
    unittest.main()
