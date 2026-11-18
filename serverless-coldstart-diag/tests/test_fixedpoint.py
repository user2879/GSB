import unittest

from coldstart_diag.fixedpoint import (
    comparison_error_bound,
    indistinguishable,
    scaled_ratio,
    single_value_error_bound,
)


class FixedPointTest(unittest.TestCase):
    def test_floor_truncation_and_ulp(self):
        # 1/3 在 scale=10000 下 floor = 3333，真值*10000=3333.333...
        self.assertEqual(scaled_ratio(1, 3, 10000), 3333)
        self.assertEqual(single_value_error_bound(), 1)
        self.assertLess(10000 / 3 - scaled_ratio(1, 3, 10000), 1)

    def test_zero_denominator(self):
        self.assertEqual(scaled_ratio(5, 0, 10000), 0)

    def test_comparison_bound(self):
        self.assertEqual(comparison_error_bound(), 2)
        self.assertTrue(indistinguishable(3333, 3335))
        self.assertFalse(indistinguishable(3333, 3336))

    def test_power_of_ten_scale(self):
        from coldstart_diag.errors import SpecError
        from coldstart_diag.fixedpoint import power_of_ten
        self.assertEqual(power_of_ten(100000), 5)
        with self.assertRaises(SpecError):
            power_of_ten(7)


if __name__ == "__main__":
    unittest.main()
