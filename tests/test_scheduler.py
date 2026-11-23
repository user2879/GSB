from __future__ import annotations

import unittest

from posdiag.scheduler import E_POINT_BUDGET_PENDING, E_SAMPLE_BUDGET_EXHAUSTED, build_budget_plan
from tests.helpers import make_config, make_sample


class SchedulerTests(unittest.TestCase):
    def test_sample_quota_exhaustion_charges_quota_and_continues(self) -> None:
        config = make_config(batch_point_budget=2, global_mac_budget=100000)
        oversized = make_sample("s0001", key_positions=(1,))
        normal = make_sample("s0002", key_positions=())
        plan = build_budget_plan(config, [normal, oversized])
        self.assertEqual([task.sample.sample_id for task in plan.tasks], ["s0001", "s0002"])
        exhausted, scheduled = plan.tasks
        self.assertEqual(exhausted.status, "budget_exhausted")
        self.assertEqual(exhausted.error_code, E_SAMPLE_BUDGET_EXHAUSTED)
        self.assertEqual(exhausted.charged_points, 2)
        self.assertEqual(scheduled.status, "scheduled")
        self.assertEqual(plan.deduction_sequence[0]["charged_points"], 2)

    def test_global_budget_exhaustion_preserves_sorted_pending_queue(self) -> None:
        config = make_config(dimension=4, global_mac_budget=4)
        samples = [
            make_sample("s0003", dimension=4, key_positions=()),
            make_sample("s0002", dimension=4, key_positions=()),
            make_sample("s0001", dimension=4, key_positions=()),
        ]
        plan = build_budget_plan(config, samples)
        self.assertEqual(plan.global_point_capacity, 1)
        self.assertEqual([task.sample.sample_id for task in plan.tasks], ["s0001", "s0002", "s0003"])
        self.assertEqual([task.status for task in plan.tasks], ["scheduled", "budget_pending", "budget_pending"])
        self.assertEqual([task.error_code for task in plan.pending_queue], [E_POINT_BUDGET_PENDING] * 2)
        self.assertEqual([task.sample.sample_id for task in plan.pending_queue], ["s0002", "s0003"])
        self.assertEqual(plan.remaining_points, 0)
        self.assertEqual(plan.consumed_points, 1)

    def test_budget_plan_is_order_independent_after_sorting(self) -> None:
        config = make_config()
        samples = [make_sample(f"s{index:04d}") for index in (4, 2, 3, 1)]
        first = build_budget_plan(config, samples)
        second = build_budget_plan(config, list(reversed(samples)))
        self.assertEqual(first.as_dict(), second.as_dict())


if __name__ == "__main__":
    unittest.main()
