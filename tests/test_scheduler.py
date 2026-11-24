from __future__ import annotations

import unittest
from dataclasses import replace

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry
from matgen.loader import load_input
from matgen.scheduler import PointBudgetScheduler, WorkItem

from tests.helpers import BASIC


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)

    def _build(self, concurrency: int, budget: int):
        config = replace(
            self.loaded.config,
            concurrency=concurrency,
            point_budget=budget,
        )
        return MaterialEngine(CacheStore(), InjectionRegistry()).build(
            config,
            self.loaded.materials,
            self.loaded.textures,
        )

    def test_budget_deduction_is_concurrency_independent(self) -> None:
        serial = self._build(1, 512)
        parallel = self._build(4, 512)
        serial_sequence = [event["deduction_sequence"] for event in serial.scheduler_events]
        parallel_sequence = [event["deduction_sequence"] for event in parallel.scheduler_events]
        self.assertEqual(serial_sequence, parallel_sequence)
        self.assertEqual(serial.final_digest, parallel.final_digest)

    def test_pending_budget_marker_and_completion(self) -> None:
        result = self._build(2, 128)
        pending = [event for event in result.scheduler_events if event.get("code") == "E_POINT_BUDGET_PENDING"]
        self.assertGreater(len(pending), 0)
        self.assertEqual(result.scheduler_events[-1]["status"], "complete")
        self.assertEqual(result.scheduler_events[-1]["pending_points"], 0)
        self.assertTrue(all(event["consumed"] <= event["budget"] for event in result.scheduler_events))

    def test_scheduler_cover_all_pixels_with_small_budget(self) -> None:
        scheduler = PointBudgetScheduler(7)
        scheduler.add(
            [
                WorkItem(
                    stage="noise",
                    stage_rank=0,
                    layer_index=0,
                    row=0,
                    col_start=0,
                    col_end=32,
                    width=32,
                )
            ]
        )
        coverage = []
        while scheduler.pending:
            for item in scheduler.take_batch():
                coverage.extend(range(item.col_start, item.col_end))
        self.assertEqual(coverage, list(range(32)))


if __name__ == "__main__":
    unittest.main()
