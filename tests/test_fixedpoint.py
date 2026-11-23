from __future__ import annotations

import unittest
from decimal import Decimal
from fractions import Fraction

from config_arbiter.fixedpoint import (
    I64_MAX,
    I64_MIN,
    add_i64,
    comparison_error_bound,
    midpoint_trunc,
    quantize,
    quantize_weight,
    saturate_i64,
    tolerance_fixed,
)


class FixedPointTests(unittest.TestCase):
    def test_quantize_positive_and_negative_toward_zero(self) -> None:
        positive = quantize(Decimal("12.3456"), 3)
        negative = quantize(Decimal("-12.3456"), 3)
        self.assertEqual(positive.fixed, 12345)
        self.assertEqual(negative.fixed, -12345)
        self.assertEqual(positive.error, Fraction(3, 5))
        self.assertEqual(negative.error, Fraction(3, 5))
        self.assertEqual(positive.error_upper_fixed_units, 1)
        self.assertEqual(negative.error_upper_fixed_units, 1)

    def test_quantize_saturates_to_signed_int64(self) -> None:
        high = quantize(Decimal("1e30"), 3)
        low = quantize(Decimal("-1e30"), 3)
        self.assertEqual(high.fixed, I64_MAX)
        self.assertEqual(low.fixed, I64_MIN)
        self.assertTrue(high.saturated)
        self.assertTrue(low.saturated)
        self.assertGreater(high.error_upper_fixed_units, 1)

    def test_integer_operations_and_weights_are_saturating(self) -> None:
        self.assertEqual(add_i64(I64_MAX, 1), I64_MAX)
        self.assertEqual(saturate_i64(I64_MIN - 1), I64_MIN)
        self.assertEqual(quantize_weight(3, 3), 3000)
        self.assertEqual(tolerance_fixed(20, 3), 20000)
        self.assertEqual(comparison_error_bound(20, 3, 1), 20002)
        self.assertEqual(midpoint_trunc(-3, 0), -1)
        self.assertEqual(midpoint_trunc(1, 2), 1)


if __name__ == "__main__":
    unittest.main()
