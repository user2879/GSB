from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quantcalib.errors import E_BATCH_BUDGET_PENDING, E_QUOTA_EXCEEDED
from quantcalib.loading import load_config, load_weights
from quantcalib.scheduler import plan_budget

from tests.helpers import base_config, write_yaml


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weights = self.root / "weights.jsonl"
        rows = []
        for name in ("z.layer", "a.layer", "m.layer"):
            rows.append(
                {
                    "layer_name": name,
                    "shape": [8],
                    "weights": [float(i + 1) for i in range(8)],
                    "content_version": 1,
                    "group_granularity": 4,
                }
            )
        self.weights.write_text(
            "".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _config(self, **scheduling):
        path = self.root / "config.yaml"
        write_yaml(path, base_config(**scheduling))
        failures = __import__("quantcalib.errors", fromlist=["FailureList"]).FailureList()
        return load_config(path, failures), failures

    def test_order_invariance_and_worker_invariance(self) -> None:
        failures = __import__("quantcalib.errors", fromlist=["FailureList"]).FailureList()
        layers = load_weights(self.weights, failures)
        config_a, _ = self._config(thread_pool_workers=1)
        config_b, _ = self._config(thread_pool_workers=8)
        plan_a = plan_budget(layers, config_a)
        plan_b = plan_budget(list(reversed(layers)), config_b)
        self.assertEqual(plan_a.consumption_sequence, plan_b.consumption_sequence)
        self.assertEqual(plan_a.batch_events, plan_b.batch_events)
        self.assertEqual([item.layer_name for item in plan_a.scheduled], [item.layer_name for item in plan_b.scheduled])

    def test_per_layer_quota_aborts_layer(self) -> None:
        config, failures = self._config(per_layer_calibration_quota=4)
        layers = load_weights(self.weights, failures)
        plan = plan_budget(layers, config)
        self.assertEqual(plan.global_consumed, 12)
        self.assertTrue(plan.pending)
        self.assertEqual({item.code for item in plan.pending}, {E_QUOTA_EXCEEDED})
        self.assertEqual(len(plan.layer_aborted), 3)

    def test_global_budget_leaves_tail_pending(self) -> None:
        config, failures = self._config(global_calibration_point_budget=8)
        layers = load_weights(self.weights, failures)
        plan = plan_budget(layers, config)
        self.assertEqual(plan.global_consumed, 8)
        self.assertEqual(len(plan.scheduled), 2)
        self.assertTrue(plan.pending)
        self.assertEqual({item.code for item in plan.pending}, {E_BATCH_BUDGET_PENDING})

    def test_batch_budget_chunks_never_exceed_limit(self) -> None:
        config, failures = self._config(batch_quantization_point_budget=3)
        layers = load_weights(self.weights, failures)
        plan = plan_budget(layers, config)
        self.assertTrue(plan.batch_events)
        for batch in plan.batch_events:
            self.assertLessEqual(batch["points"], 3)
            self.assertEqual(batch["points"], sum(chunk["points"] for chunk in batch["chunks"]))
        self.assertEqual(
            sum(batch["points"] for batch in plan.batch_events),
            plan.global_consumed,
        )


if __name__ == "__main__":
    unittest.main()
