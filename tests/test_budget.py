from __future__ import annotations

import math
import unittest

from volumetric_editor.errors import BusinessError, ErrorCode
from volumetric_editor.rendering import plan_tiles
from volumetric_editor.scene import generate_scene
from volumetric_editor.scheduling import (
    SchedulerConfig,
    estimate_task_cost,
    estimate_task_memory,
    execute_tiles,
)


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(seed=9191, width=32, height=24)
        cls.specs = plan_tiles(32, 24, 8)
        cls.max_memory = max(estimate_task_memory(cls.scene, spec) for spec in cls.specs)
        cls.max_cost = max(estimate_task_cost(cls.scene, spec) for spec in cls.specs)

    def test_memory_and_compute_pressure_degrade_parallelism(self) -> None:
        config = SchedulerConfig(
            thread_limit=4,
            queue_memory_budget_bytes=math.ceil(self.max_memory * 3.1),
            compute_budget_units=math.ceil(self.max_cost * 2.5),
        )
        result = execute_tiles(self.scene, self.specs, config)
        report = result.report
        self.assertLess(report["effective_threads"], 4)
        self.assertTrue(report["degradation_events"])
        self.assertTrue(any(event["reason"] == "QUEUE_MEMORY_LIMIT" for event in report["degradation_events"]))
        self.assertTrue(any(event["reason"] == "COMPUTE_BUDGET_LIMIT" for event in report["degradation_events"]))
        self.assertEqual(report["completed_task_count"], len(self.specs))
        self.assertEqual(sorted(result.tiles), sorted(spec.key for spec in self.specs))
        for event in report["degradation_events"]:
            self.assertIn("before", event)
            self.assertIn("after", event)

    def test_serial_fallback_still_produces_every_tile_once(self) -> None:
        config = SchedulerConfig(
            thread_limit=4,
            queue_memory_budget_bytes=math.ceil(self.max_memory * 12),
            compute_budget_units=math.ceil(self.max_cost * 1.5),
        )
        result = execute_tiles(self.scene, self.specs, config)
        self.assertTrue(result.report["serialized"])
        self.assertEqual(result.report["effective_threads"], 1)
        self.assertTrue(any(event["reason"] == "SERIAL_FALLBACK" for event in result.report["degradation_events"]))
        self.assertEqual(len(result.tiles), len(self.specs))
        self.assertEqual(set(result.tiles), {spec.key for spec in self.specs})
        self.assertTrue(all(tile.status == "COMPLETED_SERIAL" for tile in result.tiles.values()))
        self.assertEqual(result.report["completed_task_count"], len(self.specs))

    def test_zero_budget_returns_denied(self) -> None:
        config = SchedulerConfig(
            thread_limit=2,
            queue_memory_budget_bytes=0,
            compute_budget_units=100_000,
        )
        with self.assertRaises(BusinessError) as context:
            execute_tiles(self.scene, self.specs, config)
        self.assertEqual(context.exception.code, ErrorCode.BUDGET_DENIED)

    def test_budget_rejected_tasks_are_traceable(self) -> None:
        config = SchedulerConfig(
            thread_limit=2,
            queue_memory_budget_bytes=1,
            compute_budget_units=100_000,
        )
        with self.assertRaises(BusinessError) as context:
            execute_tiles(self.scene, self.specs, config)
        error = context.exception
        self.assertEqual(error.code, ErrorCode.BUDGET_DENIED)
        denied = error.details["budget_report"]["denied_tasks"]
        self.assertEqual(len(denied), len(self.specs))
        self.assertTrue(all(item["code"] == "E_BUDGET_DENIED" for item in denied))


if __name__ == "__main__":
    unittest.main()
