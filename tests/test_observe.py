import unittest

from chaoskit.observe import RequestRecord, aggregate, nearest_rank_quantile


def records(success_pattern, latencies):
    return [
        RequestRecord(timestamp_ms=i, phase="active", success=ok, latency_ms=latency)
        for i, (ok, latency) in enumerate(zip(success_pattern, latencies))
    ]


class ObservationTest(unittest.TestCase):
    def test_fixed_point_rate_and_bounds(self):
        data = records([True] * 3 + [False] * 2, [10] * 5)
        summary = aggregate(data, "active", min_samples=4)
        self.assertFalse(summary.insufficient)
        self.assertEqual(summary.success_rate_bps, 6000)
        self.assertEqual(summary.rate_quant_error_bps, 2000)
        self.assertEqual(summary.p50_latency_ms, 10)
        self.assertEqual(summary.latency_quant_error_ms, 1)

    def test_nearest_rank_quantile(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(nearest_rank_quantile(values, 0.5), 5)
        self.assertEqual(nearest_rank_quantile(values, 0.95), 10)

    def test_insufficient_samples_never_fabricated(self):
        data = records([True], [5])
        summary = aggregate(data, "active", min_samples=4)
        self.assertTrue(summary.insufficient)
        self.assertIsNone(summary.success_rate_bps)
        self.assertIsNone(summary.p95_latency_ms)
        self.assertEqual(summary.success_count, 1)
        self.assertEqual(summary.failure_count, 0)


if __name__ == "__main__":
    unittest.main()
