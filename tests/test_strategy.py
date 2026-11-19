from __future__ import annotations

import unittest

from tactical_path.models import Budgets, Query
from tactical_path.strategy import StrategyPlanner

from .helpers import bundle, config


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = config()
        cls.small = bundle("small")

    def test_three_tiers_select_three_strategies(self) -> None:
        selected = {}
        for tier in ("small", "medium", "large"):
            loaded = self.small if tier == "small" else bundle(tier)
            planner = StrategyPlanner(self.cfg)
            plan = planner.plan(
                loaded.map_data, loaded.queries, self.cfg.budgets[tier]
            )
            selected[tier] = {decision.strategy for decision in plan.decisions.values()}
        self.assertEqual(selected["small"], {"region_flow"})
        self.assertEqual(selected["medium"], {"dense_astar"})
        self.assertEqual(selected["large"], {"chunk_jump"})

    def test_same_input_reuses_all_decisions(self) -> None:
        planner = StrategyPlanner(self.cfg)
        budget = self.cfg.budgets["small"]
        first = planner.plan(self.small.map_data, self.small.queries, budget)
        second = planner.plan(self.small.map_data, self.small.queries, budget)
        self.assertEqual(len(first.recomputed), len(self.small.queries))
        self.assertEqual(second.recomputed, ())
        self.assertEqual(len(second.reused), len(self.small.queries))
        self.assertTrue(
            all(
                first.decisions[key].to_dict() == second.decisions[key].to_dict()
                for key in first.decisions
            )
        )

    def test_query_distribution_change_only_replans_affected_group(self) -> None:
        planner = StrategyPlanner(self.cfg)
        budget = self.cfg.budgets["small"]
        planner.plan(self.small.map_data, self.small.queries, budget)
        first = self.small.queries[0]
        extra = Query(
            id="q-small-extra",
            tier="small",
            unit_id=first.unit_id,
            start=first.start,
            target=first.target,
            mobility=first.mobility,
        )
        changed = planner.plan(
            self.small.map_data, self.small.queries + [extra], budget
        )
        self.assertIn(extra.id, changed.recomputed)
        self.assertGreater(len(changed.recomputed), 1)
        self.assertLess(len(changed.recomputed), len(self.small.queries) + 1)
        self.assertGreater(len(changed.reused), 0)

    def test_budget_change_replans_affected_tier(self) -> None:
        planner = StrategyPlanner(self.cfg)
        budget = self.cfg.budgets["small"]
        planner.plan(self.small.map_data, self.small.queries, budget)
        changed_budget = Budgets(
            tier=budget.tier,
            query_time_ms=budget.query_time_ms,
            query_memory_mib=budget.query_memory_mib,
            total_memory_mib=budget.total_memory_mib,
            max_threads=1,
            precompute_time_ms=budget.precompute_time_ms,
        )
        changed = planner.plan(
            self.small.map_data, self.small.queries, changed_budget
        )
        self.assertEqual(len(changed.recomputed), len(self.small.queries))
        self.assertEqual(changed.reused, ())

