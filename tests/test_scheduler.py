from __future__ import annotations

import time
import unittest

from edge_ledger.config import parse_config
from edge_ledger.scheduler import ConcurrencyProbe, RenderTask, run_thread_pool, schedule_samples


class SchedulerTests(unittest.TestCase):
    def _config(self, budget: int, quota: int, workers: int = 4):
        config, failures = parse_config(
            {
                "schema_version": 1,
                "scheduler": {
                    "max_workers": workers,
                    "render_point_budget": budget,
                    "source_quota_points": quota,
                    "points_per_render": 1,
                },
            }
        )
        self.assertFalse(failures)
        return config

    def test_quota_and_budget_are_stable(self) -> None:
        entries = [
            ("shard-a", "A1"),
            ("shard-a", "A2"),
            ("shard-a", "A3"),
            ("shard-b", "B1"),
            ("shard-b", "B2"),
            ("shard-b", "B3"),
        ]
        result = schedule_samples(entries, self._config(budget=3, quota=2))
        self.assertEqual([task.sample_id for task in result.scheduled], ["A1", "A2", "B1"])
        self.assertEqual(
            [item.sample_id for item in result.pending], ["B2"]
        )
        self.assertEqual(
            [failure.target_id for failure in result.quota_failures], ["A3", "B3"]
        )
        self.assertTrue(all(len(task.sort_key) == 71 for task in result.scheduled))

    def test_prior_pending_is_served_first(self) -> None:
        entries = [("shard-a", "A1"), ("shard-b", "B1"), ("shard-b", "B2")]
        prior = [{"sample_id": "B2", "sort_key": "old"}]
        result = schedule_samples(entries, self._config(budget=1, quota=10), prior)
        self.assertEqual([task.sample_id for task in result.scheduled], ["B2"])
        self.assertEqual(result.scheduled[0].priority_source, "prior_pending")
        self.assertEqual([item.sample_id for item in result.pending], ["A1", "B1"])

    def test_thread_pool_respects_limit_and_preserves_input_order(self) -> None:
        tasks = list(range(12))
        probe = ConcurrencyProbe()

        def worker(value: int) -> int:
            time.sleep(0.01)
            return value * value

        results = run_thread_pool(tasks, worker, 3, probe)
        self.assertEqual(results, [value * value for value in tasks])
        self.assertLessEqual(probe.max_active, 3)
        self.assertGreaterEqual(probe.max_active, 1)


if __name__ == "__main__":
    unittest.main()

