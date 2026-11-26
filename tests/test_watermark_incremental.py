import unittest

from shard_diag.impact import event_edit_impact
from shard_diag.spec import Event, EventStream, NodeSpec, ShardSpec, Spec
from shard_diag.watermark import (
    budget_candidate_judgement,
    compare_fixed_water,
    global_budget_attribution,
)


def spec():
    node = NodeSpec("n1", "127.0.0.1", 41000, 2, 2, 1, 1)
    shards = (
        ShardSpec("a", (), 8, 3, 10),
        ShardSpec("b", ("a",), 8, 2, 1),
    )
    return Spec(1, (node,), shards, 10, None)


class WatermarkTests(unittest.TestCase):
    def test_global_error_bound_and_participation(self):
        report = global_budget_attribution([("a", 3), ("b", 2)], 10, 1_000_000)
        self.assertEqual(report.used_units, 5)
        self.assertEqual(len(report.participants), 2)
        self.assertLessEqual(report.carry_q, report.error_bound_q_inclusive)
        self.assertEqual(report.error_bound_q_exclusive, 2)
        self.assertLess(report.aggregate_q - report.sum_of_floors_q, report.error_bound_q_exclusive)

    def test_budget_judgement_marks_boundary(self):
        judgement = budget_candidate_judgement(used_units=8, candidate_units=2, budget=10, scale=1_000_000)
        self.assertTrue(judgement["exact_admissible"])
        self.assertEqual(judgement["exact_slack_units"], 0)
        self.assertTrue(judgement["error_affected"])

    def test_compare_water_within_two_ulp_is_marked(self):
        close = compare_fixed_water(100, 101, error_bound_q=2)
        self.assertTrue(close["error_affected"])
        far = compare_fixed_water(0, 10, error_bound_q=2)
        self.assertFalse(far["error_affected"])


class IncrementalImpactTests(unittest.TestCase):
    def test_priority_edit_has_no_aggregate_recompute(self):
        current_spec = spec()
        old = EventStream((Event(1, "priority", {"shard_id": "b", "priority": 1}),))
        new = EventStream((Event(1, "priority", {"shard_id": "b", "priority": 99}),))
        impact = event_edit_impact(current_spec, old, new)
        self.assertEqual(impact["first_changed_seq"], 1)
        self.assertEqual(impact["aggregate_recompute_shards"], [])
        self.assertIn("b", impact["state_affected_shards"])


if __name__ == "__main__":
    unittest.main()
