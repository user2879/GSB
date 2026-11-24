from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from columnar_diag.models import BudgetConfig
from columnar_diag.planner import build_plans
from columnar_diag.sample import build_sample_spec
from columnar_diag.scan import execute_tasks
from columnar_diag.spec import parse_spec
from columnar_diag.workspace import Workspace


class PlannerTests(unittest.TestCase):
    def test_whitespace_and_parentheses_have_same_plan_digest(self) -> None:
        original = build_sample_spec().to_dict()
        altered = parse_spec(original).to_dict()
        altered["queries"][0]["where"] = " (( amount >= 10 )) AND ((region IN (3,1)))"
        plan_a = build_plans(parse_spec(original)).plan_map["q1"]
        plan_b = build_plans(parse_spec(altered)).plan_map["q1"]
        self.assertEqual(plan_a.digest, plan_b.digest)
        self.assertEqual(plan_a.candidate_blocks, plan_b.candidate_blocks)

    def test_pruning_and_pushdown(self) -> None:
        bundle = build_plans(build_sample_spec())
        q1 = bundle.plan_map["q1"]
        self.assertIn("qty", q1.pruned_columns)
        self.assertIn("channel", q1.pruned_columns)
        self.assertIn("amount@rg1", q1.candidate_blocks)
        q4 = bundle.plan_map["q4"]
        self.assertEqual(q4.candidate_blocks, ())
        self.assertTrue(any("no_predicate_match" in item for item in q4.basis))

    def test_structured_range_predicate(self) -> None:
        raw = build_sample_spec().to_dict()
        raw["queries"][0]["where"] = {
            "op": "range",
            "column": "amount",
            "low": 10,
            "high": 20,
        }
        plan = build_plans(parse_spec(raw)).plan_map["q1"]
        self.assertEqual(plan.predicate_columns, ("amount",))
        self.assertIn("amount@rg0", plan.candidate_blocks)
        self.assertIn("amount@rg1", plan.candidate_blocks)
        self.assertIn("amount@rg2", plan.candidate_blocks)


class SchedulerTests(unittest.TestCase):
    def test_thread_pool_respects_all_three_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(4, 128, 3),
                overwrite=True,
            )
            try:
                budget = BudgetConfig(
                    global_parallelism=3,
                    column_memory_budget_bytes=100,
                    segment_concurrency=2,
                )
                report = execute_tasks(
                    workspace.bundle.tasks,
                    bundle=workspace.bundle,
                    image=workspace.image,
                    budget=budget,
                    delay_ms=4.0,
                )
                metrics = report.schedule.metrics
                self.assertLessEqual(metrics["peak_global_tasks"], 3)
                self.assertTrue(
                    all(
                        count <= budget.segment_concurrency
                        for count in metrics["peak_segment_tasks"].values()
                    )
                )
                self.assertTrue(
                    all(
                        usage <= budget.column_memory_budget_bytes
                        for usage in metrics["peak_column_memory_bytes"].values()
                    )
                )
                self.assertGreater(metrics["max_queue_wait_ms"], 0.0)
                for outcome in report.schedule.outcomes:
                    self.assertGreaterEqual(outcome.wait_ended_ms, outcome.queue_started_ms)
                    self.assertGreaterEqual(outcome.run_started_ms, outcome.wait_ended_ms)
                    self.assertGreaterEqual(outcome.run_ended_ms, outcome.run_started_ms)
            finally:
                workspace.close()


if __name__ == "__main__":
    unittest.main()
