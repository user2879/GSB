import unittest
from fractions import Fraction

from graymirror.engine.errors import E_GRID, GrayError
from graymirror.engine.fixedpoint import (
    GRID_BASIS, QuantizedRatio, complement_fixed, grid_step,
    independent_sum_deviation, quantize, scale_from_exponent,
)
from graymirror.engine.hashing import stable_bucket


class FixedPointTests(unittest.TestCase):
    def test_scale_exponent_validation(self):
        self.assertEqual(scale_from_exponent(0), 1)
        self.assertEqual(scale_from_exponent(6), 1_000_000)
        for bad in (-1, 7):
            with self.assertRaises(GrayError) as ctx:
                scale_from_exponent(bad)
            self.assertEqual(ctx.exception.code, E_GRID)

    def test_floor_quantization_and_error_bound(self):
        scale = 100  # k=2, grid step 100 per-10000
        q = QuantizedRatio(3333, scale, quantize(3333, scale))
        self.assertEqual(q.fixed, 33)
        self.assertEqual(q.effective_per_10000, Fraction(3300))
        self.assertEqual(q.truncation_error, Fraction(33))
        self.assertEqual(q.error_bound, grid_step(scale))
        self.assertLess(q.truncation_error, q.error_bound)

    def test_exact_quantization_at_k4(self):
        q = QuantizedRatio(3333, 10_000, quantize(3333, 10_000))
        self.assertEqual(q.fixed, 3333)
        self.assertEqual(q.truncation_error, Fraction(0))

    def test_integer_bucket_comparison_equivalent_to_threshold(self):
        scale = 100
        q = QuantizedRatio(3333, scale, quantize(3333, scale))
        threshold = q.effective_per_10000  # 3300/10000
        for bucket in (0, 3299, 3300, 9999, 10000):
            self.assertEqual(
                q.contains_bucket(bucket),
                Fraction(bucket) < threshold)

    def test_complement_partitions_full_scale(self):
        scale = 100
        qg = quantize(3333, scale)
        self.assertEqual(qg + complement_fixed(qg, scale), scale)
        # independent floors can miss the full scale by at most one unit
        deviation = independent_sum_deviation(3333, 6667, scale)
        self.assertIn(deviation, (0, 1))

    def test_stable_bucket_range_and_determinism(self):
        for key in ("a", "client-1", "中文键"):
            bucket = stable_bucket("route", "r1", key)
            self.assertIsInstance(bucket, int)
            self.assertGreaterEqual(bucket, 0)
            self.assertLessEqual(bucket, GRID_BASIS)
            self.assertEqual(bucket, stable_bucket("route", "r1", key))
        self.assertNotEqual(
            stable_bucket("route", "r1", "k"),
            stable_bucket("mirror", "r1", "k"))


if __name__ == "__main__":
    unittest.main()
