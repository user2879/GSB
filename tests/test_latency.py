from __future__ import annotations

import unittest

from config_ledger.latency import (
    LatencySample,
    aggregate_stats,
    build_fetch_aggregates,
    evaluate_budget,
)


class LatencyTests(unittest.TestCase):
    def test_integer_fixed_point_mean_p95_and_bound(self) -> None:
        stats = aggregate_stats(range(1, 101))
        self.assertEqual(stats["sample_count"], 100)
        self.assertEqual(stats["mean_us"], 51)
        self.assertEqual(stats["p95_us"], 95)
        self.assertEqual(stats["max_us"], 100)
        self.assertEqual(stats["mean_fixed6_ms"], 51000)
        self.assertEqual(stats["p95_fixed6_ms"], 95000)
        self.assertEqual(stats["quantization_error_bound_us"], "1.5")
        self.assertEqual(stats["quantization_error_bound_fixed6_ms"], 1500)

    def test_over_budget_samples_remain_in_aggregates(self) -> None:
        samples = [
            LatencySample("fetch", "node-a", "a", 100, 1, 1, False, 0),
            LatencySample("fetch", "node-a", "b", 20000, 2, 22, True, 10000),
        ]
        aggregates = build_fetch_aggregates(samples, ["node-a", "node-b"])
        overall = aggregates[0]
        self.assertEqual(overall["scope"], "overall")
        self.assertEqual(overall["sample_count"], 2)
        self.assertEqual(overall["max_us"], 20000)
        node_a = aggregates[1]
        node_b = aggregates[2]
        self.assertEqual(node_a["mean_us"], 10050)
        self.assertEqual(node_a["sample_count"], 2)
        self.assertEqual(node_b["sample_count"], 0)

    def test_budget_uses_microsecond_integer_arithmetic(self) -> None:
        over, amount = evaluate_budget(12501, 12)
        self.assertTrue(over)
        self.assertEqual(amount, 501)
        over, amount = evaluate_budget(12000, 12)
        self.assertFalse(over)
        self.assertEqual(amount, 0)


if __name__ == "__main__":
    unittest.main()
