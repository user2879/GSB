import unittest

from fxpdiag.fixedpoint import (
    QuantSpec,
    Quantizer,
    exp_fixed,
    rsqrt_int,
    signed_bounds,
)


class FixedPointTests(unittest.TestCase):
    def test_quantize_dequant_roundtrip(self):
        q = Quantizer(QuantSpec(16, 8, "standard", "HALF_EVEN"))
        self.assertAlmostEqual(q.dequant(q.quantize(1.5)), 1.5)
        self.assertAlmostEqual(q.dequant(q.quantize(0.1)), 0.1015625, places=6)

    def test_rounding_directions_recorded(self):
        q = Quantizer(QuantSpec(16, 4, "standard", "TRUNC"))
        q.quantize(0.07)  # 0.07*16=1.12 -> TRUNC 向下
        directions = [event.direction for event in q.trace]
        self.assertIn("DOWN", directions)

    def test_bit_width_changes_saturation_bounds(self):
        q8 = Quantizer(QuantSpec(8, 4, "coarse", "HALF_EVEN"))
        q16 = Quantizer(QuantSpec(16, 4, "coarse", "HALF_EVEN"))
        big = 20.0
        self.assertNotEqual(q8.quantize(big), q16.quantize(big))
        self.assertEqual(signed_bounds(8, 0)[1], 127)
        self.assertEqual(signed_bounds(16, 0)[1], 32767)

    def test_guard_bits_tier_bounds(self):
        coarse = Quantizer(QuantSpec(8, 4, "coarse", "HALF_EVEN"))
        fine = Quantizer(QuantSpec(8, 4, "fine", "HALF_EVEN"))
        self.assertLess(coarse.qmax, fine.qmax)

    def test_exp_approx_monotone_and_range(self):
        q = Quantizer(QuantSpec(32, 12, "standard", "HALF_EVEN"))
        one = 1 << 12
        out0 = exp_fixed(q.quantize(0.0), q, 0)
        out1 = exp_fixed(q.quantize(1.0), q, 1)
        self.assertAlmostEqual(q.dequant(out0), 1.0, places=3)
        self.assertGreater(out1, one)
        self.assertLess(abs(q.dequant(out1) - 2.71828), 0.05)

    def test_rsqrt_int(self):
        f = 8
        q = Quantizer(QuantSpec(32, f, "standard", "HALF_EVEN"))
        q4 = q.quantize(4.0)
        r = rsqrt_int(q4, f)
        self.assertAlmostEqual(q.dequant(r), 0.5, places=2)

    def test_mul_division_fixed(self):
        q = Quantizer(QuantSpec(32, 10, "standard", "HALF_EVEN"))
        a = q.quantize(1.25)
        b = q.quantize(2.0)
        product = q.mul(a, b, "test", 0)
        self.assertAlmostEqual(q.dequant(product), 2.5, places=2)
        ratio = q.div(b, a, "test", 1)
        self.assertAlmostEqual(q.dequant(ratio), 1.6, places=2)


if __name__ == "__main__":
    unittest.main()


class RoundingModeTests(unittest.TestCase):
    def test_named_rounding_modes_change_integer(self):
        value = 0.101  # *16 = 1.616，TRUNC/FLOOR 与 HALF_* 必然不同
        results = {}
        for mode in ("HALF_EVEN", "HALF_UP", "TRUNC", "FLOOR"):
            quantizer = Quantizer(QuantSpec(16, 4, "standard", mode))
            results[mode] = quantizer.quantize(value)
        self.assertEqual(results["TRUNC"], results["FLOOR"])
        self.assertNotEqual(results["TRUNC"], results["HALF_UP"])
        self.assertEqual(results["HALF_EVEN"], results["HALF_UP"])

    def test_negative_half_up_rounds_away_from_zero(self):
        from fxpdiag.conversion import round_half_up_even_negative
        self.assertEqual(round_half_up_even_negative(1.5), 2)
        self.assertEqual(round_half_up_even_negative(-1.5), -2)
        self.assertEqual(round_half_up_even_negative(1.4), 1)
