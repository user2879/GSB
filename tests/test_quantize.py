import unittest

from cfgpipe.errors import SpecError
from cfgpipe.quantize import (INT64_MAX, INT64_MIN, aggregation_part,
                              chain_error_bounds, derived_value, quantize,
                              sat, tdiv)


class TestQuantize(unittest.TestCase):
    def test_truncation_toward_zero(self):
        self.assertEqual(quantize(1.23456789, 3), 1234)
        self.assertEqual(quantize(-1.23456789, 3), -1234)  # 向零而非向下
        self.assertEqual(quantize("2.5", 0), 2)
        self.assertEqual(quantize(-2.5, 0), -2)

    def test_saturation(self):
        self.assertEqual(quantize(10 ** 30, 6), INT64_MAX)
        self.assertEqual(quantize(-(10 ** 30), 6), INT64_MIN)
        self.assertEqual(sat(2 ** 70), INT64_MAX)
        self.assertEqual(sat(-(2 ** 70)), INT64_MIN)

    def test_scale_exp_range(self):
        for exp in range(7):
            self.assertEqual(quantize(1, exp), 10 ** exp)
        with self.assertRaises(SpecError):
            quantize(1, 7)

    def test_tdiv(self):
        self.assertEqual(tdiv(7, 2), 3)
        self.assertEqual(tdiv(-7, 2), -3)
        self.assertEqual(tdiv(0, 5), 0)

    def test_derived_value(self):
        # coeff 2.5 (2500), parent 1.5 (1500), scale 1000 -> 3750
        self.assertEqual(derived_value(2500, 1500, 3), 3750)
        self.assertEqual(derived_value(-2500, 1500, 3), -3750)

    def test_chain_error_bounds(self):
        topo = ["a", "b", "c"]
        derived_from = {"a": "", "b": "a", "c": "b"}
        coeff_q = {"b": 2500, "c": 500}
        bounds = chain_error_bounds(topo, derived_from, coeff_q, 3)
        self.assertEqual(bounds["a"], (1, 0, 1))
        # b: derived = trunc(2500*1/1000) = 2
        self.assertEqual(bounds["b"], (1, 2, 3))
        # c: derived = trunc(500*3/1000) = 1
        self.assertEqual(bounds["c"], (1, 1, 2))

    def test_aggregation_part(self):
        self.assertEqual(aggregation_part([100, 250], 8), 32)
        self.assertEqual(aggregation_part([], 8), 0)


if __name__ == "__main__":
    unittest.main()
