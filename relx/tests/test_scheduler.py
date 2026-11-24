import unittest

from relx.errors import E_FEATURE_BUDGET_OVERSIZE, E_FEATURE_BUDGET_PENDING
from relx.scheduler import BudgetScheduler, ScheduledTask


class SchedulerTest(unittest.TestCase):
    def test_pending_tasks_prioritized_next_batch(self):
        scheduler = BudgetScheduler(budget_per_batch=10)
        tasks = [ScheduledTask("S1", "entity", 6),
                 ScheduledTask("S2", "entity", 6),
                 ScheduledTask("S3", "entity", 4)]
        runnable, pending, oversize, remaining = scheduler.plan_wave(tasks)
        self.assertEqual([item.sentence_id for item in runnable], ["S1", "S3"])
        self.assertEqual(remaining, 0)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].code, E_FEATURE_BUDGET_PENDING)
        self.assertEqual(scheduler.pending[0].sentence_id, "S2")

        runnable2, pending2, _over, _rem = scheduler.plan_wave([])
        self.assertEqual([item.sentence_id for item in runnable2], ["S2"])
        self.assertEqual(pending2, [])

    def test_oversize_task_marked_and_removed(self):
        scheduler = BudgetScheduler(budget_per_batch=5)
        runnable, pending, oversize, _rem = scheduler.plan_wave(
            [ScheduledTask("S1", "relation", 9)]
        )
        self.assertEqual(runnable, [])
        self.assertEqual(pending, [])
        self.assertEqual(oversize[0].code, E_FEATURE_BUDGET_OVERSIZE)
        self.assertFalse(scheduler.has_pending())

    def test_ledger_global_consumption_consistent(self):
        scheduler = BudgetScheduler(budget_per_batch=10)
        scheduler.plan_wave([ScheduledTask("S1", "entity", 4),
                             ScheduledTask("S2", "entity", 7)])
        scheduler.plan_wave([])
        self.assertTrue(scheduler.verify_ledger())
        self.assertEqual(scheduler.global_consumed, 11)

    def test_stable_order_under_repeated_planning(self):
        scheduler = BudgetScheduler(budget_per_batch=8)
        first, _p, _o, _r = scheduler.plan_wave([
            ScheduledTask("S2", "entity", 5),
            ScheduledTask("S1", "entity", 5),
        ])
        self.assertEqual([item.sentence_id for item in first], ["S1"])
        second, pending2, _o2, remaining2 = scheduler.plan_wave([
            ScheduledTask("S3", "entity", 5)
        ])
        self.assertEqual([item.sentence_id for item in second], ["S2"])
        self.assertEqual([item.sentence_id for item in scheduler.pending], ["S3"])
        self.assertEqual(remaining2, 3)
        third, _p3, _o3, _r3 = scheduler.plan_wave([])
        self.assertEqual([item.sentence_id for item in third], ["S3"])


if __name__ == "__main__":
    unittest.main()
