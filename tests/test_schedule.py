import unittest

from quantcalib.errors import E_POINT_BUDGET_PENDING, E_QUOTA_EXCEEDED
from quantcalib.schedule.budget import BudgetScheduler, WorkUnit


class SchedulerTests(unittest.TestCase):
    def test_triple_budget_constraints_and_stable_order(self):
        units = [
            WorkUnit("b", 1, 20, "quant"),
            WorkUnit("a", 0, 30, "quant"),
            WorkUnit("a", 1, 40, "quant"),
        ]
        scheduler = BudgetScheduler(
            max_workers=2, global_budget=100, per_layer_quota=50,
            batch_budget=1000)
        planned, pending, failures = scheduler.plan(units)
        self.assertEqual([(u.layer, u.group_index) for u in planned],
                         [("a", 0), ("b", 1)])
        self.assertIn(E_QUOTA_EXCEEDED, {f["code"] for f in failures})

    def test_global_budget_pending(self):
        units = [WorkUnit("a", 0, 60, "quant"),
                 WorkUnit("b", 0, 60, "quant")]
        scheduler = BudgetScheduler(
            max_workers=2, global_budget=70, per_layer_quota=100,
            batch_budget=1000)
        planned, pending, failures = scheduler.plan(units)
        self.assertEqual(len(planned), 1)
        self.assertEqual(len(pending), 1)
        self.assertEqual(failures[-1]["code"], E_POINT_BUDGET_PENDING)

    def test_worker_count_never_exceeds_configuration(self):
        units = [WorkUnit(f"l{i}", 0, 1, "quant") for i in range(10)]
        scheduler = BudgetScheduler(
            max_workers=3, global_budget=1000, per_layer_quota=1000,
            batch_budget=1000)
        outcome, results = scheduler.run(units, lambda unit: unit.layer)
        self.assertLessEqual(outcome.max_workers_observed, 3)
        # Results returned in planned (lexicographic) order.
        self.assertEqual(results, [f"l{i}" for i in range(10)])


if __name__ == "__main__":
    unittest.main()
