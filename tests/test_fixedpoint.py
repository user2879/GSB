import unittest

from arbplatform.errors import FixedPointError
from arbplatform.fixedpoint import INT64_MAX, FixedPoint


class FixedPointTest(unittest.TestCase):
    def test_scale_and_truncation_towards_zero(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.quantize(3).q, 300)
        self.assertEqual(fp.quantize(3.149).q, 314)
        self.assertEqual(fp.quantize(-3.149).q, -314)
        self.assertEqual(fp.dequantize(314), "3.14")
        self.assertEqual(fp.dequantize(-5), "-0.05")

    def test_power_zero(self):
        fp = FixedPoint(0)
        self.assertEqual(fp.scale, 1)
        self.assertEqual(fp.quantize(7).q, 7)
        self.assertEqual(fp.dequantize(7), "7")

    def test_invalid_power(self):
        for bad in (-1, 4):
            with self.assertRaises(FixedPointError):
                FixedPoint(bad)

    def test_saturation(self):
        fp = FixedPoint(3)
        huge = fp.quantize(10**30)
        self.assertEqual(huge.q, INT64_MAX)
        self.assertTrue(huge.saturated)
        tiny = fp.quantize(-(10**30))
        self.assertEqual(tiny.q, -INT64_MAX - 1)
        self.assertTrue(tiny.saturated)

    def test_integer_multiply_divide_trunc(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.div_trunc(700, 300), 2)
        self.assertEqual(fp.div_trunc(-700, 300), -2)
        self.assertEqual(fp.mul(200, 3), 600)
        with self.assertRaises(FixedPointError):
            fp.div_trunc(1, 0)

    def test_error_bounds(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.single_key_error_ulp(), 1)
        self.assertEqual(fp.comparison_bound_ulp(1), 2 + 100)
        self.assertEqual(fp.view_aggregate_bound_ulp(3, 1), 3 * 102)

    def test_float_exactness_no_binary_artifact(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.quantize(0.29).q, 29)
        self.assertEqual(fp.quantize(1.01).q, 101)


if __name__ == "__main__":
    unittest.main()
