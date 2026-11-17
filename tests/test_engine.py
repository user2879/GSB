from __future__ import annotations

import unittest

from autoscale_diag.aggregation import WindowStats, choose_path
from autoscale_diag.engine import (
    allocate_budget,
    evaluate_workload,
    simulate,
)
from autoscale_diag.spec import Observation, Rule, Workload, load_spec

from tests.helpers import SPEC_V1


def decision_at(result, minute: int, workload_id: str):
    for item in result.minutes:
        if item.minute == minute:
            return next(decision for decision in item.decisions if decision.workload_id == workload_id)
    raise AssertionError("minute not found")


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_spec(SPEC_V1)
        cls.result = simulate(cls.spec)

    def test_cold_start_and_first_expansion(self) -> None:
        minute0 = {item.workload_id: item for item in self.result.minutes[0].decisions}
        self.assertEqual(minute0["checkout"].previous_replicas, 1)
        self.assertEqual(minute0["payments"].decision_type, "expand")
        self.assertTrue(minute0["payments"].change_took_effect)
        self.assertEqual(minute0["payments"].final_replicas, 2)
        minute1 = decision_at(self.result, 1, "checkout")
        self.assertEqual(minute1.expected_replicas, 2)
        self.assertEqual(minute1.final_replicas, 2)
        self.assertEqual(minute1.decision_type, "expand")
        self.assertTrue(minute1.change_took_effect)

    def test_rule_quantization_intermediates(self) -> None:
        workload = Workload(
            id="w",
            observations=(Observation(0, 100, 10, 0),),
            rules=(Rule("r", "requests", 60, 50),),
            concurrency_per_replica=10,
            min_replicas=1,
            max_replicas=5,
            cooldown_ms=60000,
        )
        stats = WindowStats(1, 100, 0, 10, 100, 10, 0)
        evaluations, selected, expected = evaluate_workload(workload, stats)
        evaluation = evaluations[0]
        self.assertEqual(selected, "r")
        self.assertEqual(evaluation.demand_numerator, 10000)
        self.assertEqual(evaluation.demand_denominator, 3000)
        self.assertEqual(evaluation.unclamped_replicas, 4)
        self.assertEqual(evaluation.clamped_replicas, 4)
        self.assertEqual(evaluation.quantization_error_upper_bound_milli_replicas, 667)
        self.assertEqual(evaluation.quantization_error_upper_bound_metric_units, 20)
        self.assertEqual(expected, 4)

    def test_multiple_rules_choose_largest_and_lowest_id_on_tie(self) -> None:
        workload = Workload(
            id="w",
            observations=(Observation(0, 100, 100, 1),),
            rules=(Rule("z", "requests", 60, 50), Rule("a", "requests", 60, 50)),
            concurrency_per_replica=10,
            min_replicas=1,
            max_replicas=5,
            cooldown_ms=60000,
        )
        stats = WindowStats(1, 100, 1, 100, 100, 100, 10000)
        evaluations, selected, expected = evaluate_workload(workload, stats)
        self.assertEqual([item.rule_id for item in evaluations], ["a", "z"])
        self.assertEqual(selected, "a")
        self.assertEqual(expected, 4)

    def test_budget_clipping_is_order_independent(self) -> None:
        ids1 = ["checkout", "payments", "search"]
        ids2 = ["search", "checkout", "payments"]
        desired1 = {"checkout": 3, "payments": 2, "search": 4}
        desired2 = {"search": 4, "checkout": 3, "payments": 2}
        minimums = {"checkout": 1, "payments": 1, "search": 1}
        maximums = {"checkout": 4, "payments": 3, "search": 4}
        first = allocate_budget(ids1, desired1, minimums, maximums, 6)
        second = allocate_budget(ids2, desired2, minimums, maximums, 6)
        self.assertEqual(first, second)
        self.assertEqual(first.allocated_total, 6)
        self.assertLessEqual(first.allocated_total, 6)
        self.assertEqual(first.affected_workloads, tuple(sorted(first.clipped)))

    def test_budget_is_hard_constraint_every_minute(self) -> None:
        self.assertTrue(
            all(minute.budget_used <= self.spec.global_replica_budget for minute in self.result.minutes)
        )

    def test_cooldown_suppresses_change_and_traces_trigger(self) -> None:
        minute3_checkout = decision_at(self.result, 3, "checkout")
        self.assertTrue(minute3_checkout.cooldown_active)
        self.assertTrue(minute3_checkout.cooldown_suppressed)
        self.assertEqual(minute3_checkout.cooldown_code, "E_COOLDOWN")
        self.assertEqual(minute3_checkout.cooldown_trigger_minute, 1)
        self.assertEqual(minute3_checkout.final_replicas, minute3_checkout.previous_replicas)
        self.assertIn("E_COOLDOWN", minute3_checkout.cooldown_message or "")

    def test_paths_are_byte_identical_on_decisions_and_stats(self) -> None:
        full = simulate(self.spec, forced_path="full")
        incremental = simulate(self.spec, forced_path="incremental")
        self.assertEqual(full.decision_digest(), incremental.decision_digest())
        self.assertEqual(full.peak_memory_estimate_bytes, 5248)
        self.assertEqual(incremental.peak_memory_estimate_bytes, 4384)
        self.assertGreater(full.peak_memory_estimate_bytes, incremental.peak_memory_estimate_bytes)

    def test_path_selection_rule_is_deterministic(self) -> None:
        self.assertEqual(choose_path(12, 20, "auto"), "full")
        self.assertEqual(choose_path(20, 20, "auto"), "incremental")
        self.assertEqual(choose_path(20, 20, "full"), "full")
        self.assertEqual(choose_path(2, 20, "incremental"), "incremental")

    def test_budget_and_cooldown_are_co_recorded(self) -> None:
        self.assertTrue(any(minute.budget.clipped for minute in self.result.minutes))
        self.assertTrue(any(item.cooldown_suppressed for minute in self.result.minutes for item in minute.decisions))
        self.assertTrue(any(item.budget_clipped for minute in self.result.minutes for item in minute.decisions))


if __name__ == "__main__":
    unittest.main()
