import unittest

import numpy as np

from arcore.fixedpoint import (
    compute_stats,
    quantize_array,
    quantize_scalar,
    quantization_error_bound,
    ratio_comparison_error_bound,
)


class FixedPointTest(unittest.TestCase):
    def test_truncation_toward_zero(self):
        self.assertEqual(quantize_scalar(0.123456, 10000), 1234)
        self.assertEqual(quantize_scalar(-0.123456, 10000), -1234)

    def test_saturation_clips_overflow(self):
        self.assertEqual(quantize_scalar(1e9, 10000, 1000), 1000)
        self.assertEqual(quantize_scalar(-1e9, 10000, 1000), -1000)
        clipped = quantize_array([1e9, -1e9], 10000, 1000)
        self.assertTrue(np.array_equal(clipped, np.array([1000, -1000])))

    def test_error_bounds(self):
        self.assertEqual(quantization_error_bound(10000), 0.0001)
        self.assertEqual(ratio_comparison_error_bound(10000), 0.0002)

    def test_compute_stats_fields(self):
        stats = compute_stats(
            total_alerts=10, suppressed_count=2,
            correlated_alert_count=5, queue_wait_ms=[10, 20, 30],
            scale=10000, saturation_max=1_000_000)
        self.assertEqual(stats.suppressed_ratio_q, 2000)
        self.assertEqual(stats.correlation_hit_rate_q, 5000)
        self.assertEqual(stats.average_queue_ms_q, 200000)

    def test_rejects_non_power_of_ten(self):
        with self.assertRaises(ValueError):
            quantize_scalar(1.0, 500)


if __name__ == "__main__":
    unittest.main()
