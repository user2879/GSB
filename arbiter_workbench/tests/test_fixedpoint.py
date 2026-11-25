import unittest

from arbiter.errors import QuantError
from arbiter.fixedpoint import (
    FixedPoint,
    MAX_INT64,
    MIN_INT64,
    normalize_weights,
    per_key_error_bound,
    trunc_div_toward_zero,
    view_aggregate_bound,
)


class FixedPointTest(unittest.TestCase):
    def test_trunc_toward_zero(self):
        self.assertEqual(trunc_div_toward_zero(7, 2), 3)
        self.assertEqual(trunc_div_toward_zero(-7, 2), -3)
        self.assertEqual(trunc_div_toward_zero(-6, 2), -3)

    def test_quantize_truncation_direction(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.quantize("1.239").q, 123)
        self.assertEqual(fp.quantize("-1.239").q, -123)
        self.assertEqual(fp.quantize(5).q, 500)

    def test_saturation(self):
        fp = FixedPoint(3)
        huge = str(MAX_INT64)
        q = fp.quantize(huge)
        self.assertEqual(q.q, MAX_INT64)
        self.assertTrue(q.saturated)
        low = fp.quantize(str(MIN_INT64 - 5))
        self.assertEqual(low.q, MIN_INT64)
        self.assertTrue(low.saturated)

    def test_invalid_exponent(self):
        with self.assertRaises(QuantError):
            FixedPoint(4)

    def test_weight_normalization_sums_exactly(self):
        fp = FixedPoint(2)
        for weights in ([5, 3, 2], [1, 1, 1, 1], [7, 11, 13, 17, 19]):
            normalized, bound = normalize_weights(weights, fp)
            self.assertEqual(sum(normalized), fp.factor)
            self.assertEqual(bound, len(weights) - 1)
            # ordering stability: equal weights yield identical allocations
        again, _ = normalize_weights([5, 3, 2], fp)
        self.assertEqual(again, normalize_weights([5, 3, 2], fp)[0])

    def test_error_bound_three_contributions(self):
        fp = FixedPoint(2)
        latest = per_key_error_bound(fp, "latest", 3)
        self.assertEqual((latest.quantization_truncation,
                          latest.interval_boundary,
                          latest.weight_normalization), (3, 0, 0))
        interval = per_key_error_bound(fp, "interval_midpoint", 3)
        self.assertEqual(interval.interval_boundary, 1)
        weighted = per_key_error_bound(fp, "weighted_majority", 4)
        self.assertEqual(weighted.weight_normalization, 3)
        agg = view_aggregate_bound([latest, interval, weighted])
        self.assertEqual(agg.quantization_truncation, 4)
        self.assertEqual(agg.interval_boundary, 1)
        self.assertEqual(agg.weight_normalization, 3)


if __name__ == "__main__":
    unittest.main()
