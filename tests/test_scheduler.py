import tempfile
import unittest
from pathlib import Path

from colbench.sample_data import generate
from colbench.scheduler import Scheduler
from colbench.segment import build_segment
from colbench.versioning import BudgetParams
from colbench.errors import BudgetError
from colbench.workbench import Workbench


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        paths = generate(self.tmp / "samples")
        self.bench = Workbench(self.tmp)
        self.bench.build_from_specs(paths)

    def test_triple_budget_respected(self):
        params = BudgetParams(global_parallelism=2, per_column_memory=300,
                              per_segment_concurrency=1)
        plan = self.bench.plan(
            "SELECT order_id, amount FROM orders WHERE priority >= 0")
        result = Scheduler(params, self.bench.segments).run(
            plan.candidate_blocks)
        stats = result.statistics()
        self.assertLessEqual(stats["max_running_observed"], 2)
        self.assertTrue(all(load <= 1 for load in
                            stats["max_segment_running"].values()))
        self.assertTrue(all(load <= 300 for load in
                            stats["max_column_memory"].values()))
        self.assertEqual(stats["violations"], [])
        self.assertTrue(all(task.wait_ms >= 0 and task.run_ms >= 0
                            for task in result.tasks))

    def test_zero_memory_budget_rejected(self):
        with self.assertRaises(BudgetError) as ctx:
            BudgetParams(1, 0, 1).validate()
        self.assertEqual(ctx.exception.code, "E_BUDGET")

    def test_block_over_budget_rejected(self):
        params = BudgetParams(global_parallelism=2, per_column_memory=1,
                              per_segment_concurrency=1)
        plan = self.bench.plan("SELECT order_id FROM orders")
        with self.assertRaises(BudgetError) as ctx:
            Scheduler(params, self.bench.segments).run(
                plan.candidate_blocks)
        self.assertEqual(ctx.exception.code, "E_BUDGET")

    def test_queue_wait_recorded(self):
        params = BudgetParams(global_parallelism=1, per_column_memory=8192,
                              per_segment_concurrency=1)
        plan = self.bench.plan("SELECT order_id FROM orders")
        result = Scheduler(params, self.bench.segments).run(
            plan.candidate_blocks)
        self.assertGreaterEqual(sum(t.wait_ms for t in result.tasks), 0)
        self.assertEqual([t.task_id for t in result.tasks],
                         sorted(t.task_id for t in result.tasks))


if __name__ == "__main__":
    unittest.main()
