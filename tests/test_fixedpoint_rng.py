import unittest

from fxrender.fixedpoint import (FpSpec, fixed_add, fixed_mul, quantize,
                                 rescale, validate_fp_enum)
from fxrender.rng import DeterministicRNG


class FixedPointTests(unittest.TestCase):
    def test_quantize_round_half_away(self):
        spec = FpSpec(32, "standard")
        self.assertEqual(quantize(0.5, spec), spec.scale // 2)
        self.assertEqual(quantize(-0.5, spec), -(spec.scale // 2))

    def test_enum_rejection(self):
        with self.assertRaises(ValueError):
            validate_fp_enum(24, "standard")
        with self.assertRaises(ValueError):
            FpSpec(32, "ultra")

    def test_width_tier_changes_integer(self):
        coarse = FpSpec(32, "coarse")
        fine = FpSpec(32, "fine")
        self.assertNotEqual(quantize(0.123456, coarse),
                            quantize(0.123456, fine))
        self.assertNotEqual(coarse.frac_bits, fine.frac_bits)

    def test_rescale_reports_residual(self):
        src = FpSpec(32, "fine")
        dst = FpSpec(32, "coarse")
        value = quantize(0.123457, src)
        new_value, residual = rescale(value, src, dst)
        self.assertIsInstance(new_value, int)
        self.assertNotEqual(residual, 0)

    def test_fixed_mul_residual(self):
        spec = FpSpec(16, "fine")
        a = quantize(0.7, spec)
        b = quantize(0.3, spec)
        product, residual = fixed_mul(a, b, spec)
        self.assertIsInstance(product, int)
        self.assertEqual(fixed_add(product, 0, spec), product)


class RNGTests(unittest.TestCase):
    def test_deterministic_and_independent(self):
        a = [DeterministicRNG("seed").next_gaussian() for _ in range(5)]
        b = [DeterministicRNG("seed").next_gaussian() for _ in range(5)]
        c = [DeterministicRNG("other").next_gaussian() for _ in range(5)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_schemes_bounded(self):
        rng = DeterministicRNG(7)
        uniforms = [rng.noise("uniform") for _ in range(100)]
        self.assertTrue(all(-1 <= v <= 1 for v in uniforms))
        lap = [DeterministicRNG(9).noise("laplacian") for _ in range(50)]
        self.assertEqual(len(lap), 50)


if __name__ == "__main__":
    unittest.main()
