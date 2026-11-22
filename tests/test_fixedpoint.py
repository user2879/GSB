import unittest
from fractions import Fraction

import numpy as np

from degprop.fixedpoint import (
    FP_MAX,
    Scaling,
    interval_add,
    judgment_bound,
    numpy_window_rates,
    path_aggregate,
    quantized_interval,
)


class FixedPointTest(unittest.TestCase):
    def test_scale_factor_and_quantum(self):
        scaling = Scaling(2)
        self.assertEqual(scaling.factor, 100)
        self.assertEqual(scaling.quantum, Fraction(1, 100))
        self.assertEqual(scaling.encode_ratio(Fraction(2, 3)), 66)  # floor

    def test_exponent_range(self):
        for exponent in range(4):
            self.assertEqual(Scaling(exponent).factor, 10**exponent)
        with self.assertRaises(ValueError):
            Scaling(4)

    def test_saturation(self):
        scaling = Scaling(3)
        self.assertEqual(scaling.encode_count(FP_MAX), FP_MAX)

    def test_single_value_error_bound(self):
        scaling = Scaling(2)
        interval = quantized_interval(Fraction(1, 3), scaling, "x")
        self.assertEqual(interval.err, 100)  # factor=100，q 在定点单位下为 1*F
        self.assertEqual(interval.intermediates[0]["truncation"], "floor")

    def test_path_aggregate_bounds(self):
        scaling = Scaling(2)
        a = quantized_interval(Fraction(1, 2), scaling, "a")
        b = quantized_interval(Fraction(2, 3), scaling, "b")
        path = path_aggregate([a, b], scaling.factor, "path")
        self.assertGreater(path.err, 0)
        self.assertTrue(any(i["op"] == "path_aggregate"
                            for i in path.intermediates))

    def test_judgment_bound(self):
        scaling = Scaling(2)
        a = quantized_interval(Fraction(50, 100), scaling, "a")
        b = quantized_interval(Fraction(50, 100), scaling, "b")
        verdict = judgment_bound(a, b, scaling.factor, "j")
        self.assertTrue(verdict["indistinguishable"])
        self.assertEqual(verdict["judgment_bound_fixed"], 200)

    def test_interval_add_and_numpy_rates(self):
        scaling = Scaling(2)
        a = quantized_interval(Fraction(1, 4), scaling, "a")
        b = quantized_interval(Fraction(1, 4), scaling, "b")
        added = interval_add(a, b, scaling.factor, "sum")
        self.assertEqual(added.value, 50)
        rates = numpy_window_rates(np.array([3, 0]), np.array([4, 0]), 100)
        self.assertEqual(rates.tolist(), [75, 0])


if __name__ == "__main__":
    unittest.main()
