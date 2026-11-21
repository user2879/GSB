import unittest
from decimal import Decimal

from workbench.fixedpoint import (
    clamp_fixed,
    decimal_to_fixed,
    fixed_to_str,
    quantize_to_grid,
    scale_factor,
)


class FixedPointTests(unittest.TestCase):
    def test_round_trip_decimal_strings(self):
        bits = 16
        # binary-exact fractions round-trip exactly
        for text in ("0", "0.25", "0.5", "0.75", "1", "0.0625"):
            fixed = decimal_to_fixed(text, bits)
            self.assertEqual(Decimal(fixed_to_str(fixed, bits)), Decimal(text))
        # other decimals land within one Q16 step
        step = Decimal(1) / Decimal(scale_factor(bits))
        for text in ("0.1", "0.3333333333", "0.02", "0.0001"):
            fixed = decimal_to_fixed(text, bits)
            self.assertLessEqual(abs(Decimal(fixed_to_str(fixed, bits)) - Decimal(text)), step)

    def test_quantization_grids_differ(self):
        bits = 16
        value = decimal_to_fixed("0.4997", bits)
        coarse = quantize_to_grid(value, "0.1", bits)
        medium = quantize_to_grid(value, "0.01", bits)
        fine = quantize_to_grid(value, "0.0001", bits)
        self.assertNotEqual(fixed_to_str(coarse), fixed_to_str(medium))
        self.assertNotEqual(fixed_to_str(medium), fixed_to_str(fine))
        # coarse step in Q16 is the rounded encoding of decimal 0.1
        self.assertEqual(coarse, 5 * decimal_to_fixed("0.1", bits))
        self.assertEqual(medium, 50 * decimal_to_fixed("0.01", bits))
        # finer grids stay within their grid step of the source
        self.assertLessEqual(abs(Decimal(fixed_to_str(fine)) - Decimal("0.4997")),
                             Decimal("0.0001"))
        self.assertLessEqual(abs(Decimal(fixed_to_str(medium)) - Decimal("0.4997")),
                             Decimal("0.01"))
        self.assertLessEqual(abs(Decimal(fixed_to_str(coarse)) - Decimal("0.4997")),
                             Decimal("0.1"))

    def test_clamp_bounds(self):
        factor = scale_factor()
        self.assertEqual(clamp_fixed(-5), 0)
        self.assertEqual(clamp_fixed(factor + 99), factor)
        self.assertEqual(clamp_fixed(factor // 2), factor // 2)

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            decimal_to_fixed(float("nan"))
        with self.assertRaises(ValueError):
            decimal_to_fixed(float("inf"))


if __name__ == "__main__":
    unittest.main()
