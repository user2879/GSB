from __future__ import annotations

import unittest

from tactical_path.engine import TierEngine
from tactical_path.errors import ErrorCode, TacticalError
from tactical_path.strategy import StrategyPlanner

from .helpers import bundle, config


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = config()
        cls.bundle = bundle("small")
        cls.plan = StrategyPlanner(cls.cfg).plan(
            cls.bundle.map_data,
            cls.bundle.queries,
            cls.cfg.budgets["small"],
        )

    def test_budget_downgrade_preserves_every_query(self) -> None:
        engine = TierEngine(self.bundle, self.cfg)
        budget = self.cfg.budgets["small"]
        initial = engine.solve_batch(self.bundle.queries, self.plan.decisions, budget)
        expected = {
            query_id: route.route_hash(self.bundle.map_data.width)
            for query_id, route in initial["routes"].items()
        }
        tight = engine.solve_batch(
            self.bundle.queries,
            self.plan.decisions,
            budget,
            force=True,
            memory_budget_mib=self.cfg.tight_total_memory_mib,
        )
        self.assertEqual(len(tight["routes"]), len(self.bundle.queries))
        self.assertEqual(
            [item["stage"] for item in tight["downgrades"]],
            ["serial", "cache_eviction", "minimal_memory"],
        )
        self.assertEqual(tight["workers_used"], 1)
        for query_id, route in tight["routes"].items():
            self.assertEqual(route.route_hash(self.bundle.map_data.width), expected[query_id])

    def test_negative_budget_is_rejected(self) -> None:
        engine = TierEngine(self.bundle, self.cfg)
        with self.assertRaises(TacticalError) as raised:
            engine.solve_batch(
                self.bundle.queries,
                self.plan.decisions,
                self.cfg.budgets["small"],
                memory_budget_mib=-1,
            )
        self.assertEqual(raised.exception.code, ErrorCode.BUDGET_DENIED)

