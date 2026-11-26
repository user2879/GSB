import unittest

from fxrender.codes import BizCode
from fxrender.config import RenderConfig
from fxrender.policy import select_policy
from fxrender.scheduler import Scheduler


class SchedulerTests(unittest.TestCase):
    def test_budget_pending_then_priority_resume(self):
        cfg = RenderConfig(batch_budget=10, points_per_step=4,
                          denoise_steps=4)
        policy = select_policy(2, cfg)
        scheduler = Scheduler(cfg, policy, frame_count=2)
        first = scheduler.run_batch()
        # 10 budget -> two steps (8 points), third step retained as pending.
        self.assertEqual(len(first.deductions), 2)
        self.assertTrue(first.pending)
        self.assertEqual(first.pending_events[0]["code"],
                         BizCode.E_BUDGET_PENDING.value)
        consumed_so_far = scheduler.total_consumed
        second = scheduler.run_batch()
        # Pending head is processed first in the fresh batch.
        self.assertTrue(second.processed)
        self.assertEqual(second.processed[0], first.pending[0])
        self.assertGreater(scheduler.total_consumed, consumed_so_far)
        # No key is deducted twice.
        keys = [tuple(d["key"]) for d in scheduler.deductions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_double_charge_across_all_batches(self):
        cfg = RenderConfig(batch_budget=6, points_per_step=3,
                          denoise_steps=5)
        policy = select_policy(1, cfg)
        scheduler = Scheduler(cfg, policy, frame_count=1)
        while any(item.phase == "denoise" for item in scheduler.todo):
            scheduler.run_batch()
        expected = cfg.denoise_steps * cfg.points_per_step
        self.assertEqual(scheduler.total_consumed, expected)
        self.assertEqual(sum(d["cost"] for d in scheduler.deductions),
                         expected)

    def test_max_batches_retains_todo(self):
        cfg = RenderConfig(batch_budget=4, points_per_step=4,
                          denoise_steps=6)
        policy = select_policy(1, cfg)
        scheduler = Scheduler(cfg, policy, frame_count=1)
        batch = scheduler.run_batch()
        self.assertEqual(len(batch.deductions), 1)
        self.assertTrue(batch.pending)
        self.assertGreater(len(scheduler.todo), 0)


if __name__ == "__main__":
    unittest.main()
