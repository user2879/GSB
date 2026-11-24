import unittest

import numpy as np

from opdiag import dmath
from opdiag.fixedpoint import quantize, ALLOWED_BIT_WIDTHS


class TestDeterministicMath(unittest.TestCase):
    def test_exp_approx_bounded_and_deterministic(self):
        x = np.linspace(-3, 3, 37, dtype=np.float32)
        y1, r1 = dmath.approx_exp(x)
        y2, r2 = dmath.approx_exp(x)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(r1, r2)
        # 与库参考值接近（近似而非恒等）
        np.testing.assert_allclose(y1, np.exp(x), rtol=1e-4, atol=1e-5)
        self.assertTrue(np.all(r1 > 0))

    def test_sqrt_approx(self):
        x = np.linspace(0.1, 9.0, 31, dtype=np.float64)
        y, r = dmath.approx_sqrt(x)
        np.testing.assert_allclose(y, np.sqrt(x), rtol=1e-10, atol=1e-12)
        self.assertEqual(int(r.flat[0]),
                         2 + dmath.SQRT_NEWTON_ITERATIONS * 3)

    def test_activations_finite_and_shape_preserving(self):
        x = np.linspace(-4, 4, 64, dtype=np.float32)
        for op, params in [("relu", {"slope": 0.1}), ("gelu", {}),
                           ("sigmoid", {"scale": 1.2}),
                           ("tanh", {"gain": 0.8})]:
            y, rounds = dmath.apply_activation(op, x, params)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertEqual(rounds.shape, x.shape)

    def test_precision_switch_changes_values(self):
        x = np.linspace(-2, 2, 24, dtype=np.float64)
        y32, _ = dmath.gelu(x.astype(np.float32))
        y64, _ = dmath.gelu(x.astype(np.float64))
        # fp32 与 fp64 结果不应逐位相同
        self.assertFalse(np.array_equal(y32.astype(np.float64), y64))


class TestFixedPoint(unittest.TestCase):
    def test_quantize_dequant_residual(self):
        x = np.random.default_rng(0).uniform(-3, 3, (4, 5)).astype(np.float32)
        q = quantize(x, 16, "uniform")
        self.assertEqual(q["q"].dtype, np.int16)
        np.testing.assert_allclose(q["dequant"],
                                   q["q"].astype(np.float32) *
                                   np.float32(q["step"]), rtol=1e-6)
        np.testing.assert_allclose(q["residual"], x - q["dequant"],
                                   rtol=1e-6)

    def test_bitwidth_and_mode_change_integers(self):
        x = np.linspace(-5, 5, 40, dtype=np.float32)
        q16 = quantize(x, 16, "uniform")
        q8 = quantize(x, 8, "pow2")
        self.assertEqual(q8["q"].dtype, np.int8)
        self.assertFalse(np.array_equal(
            q16["q"].astype(np.int32), q8["q"].astype(np.int32)))
        self.assertGreater(np.max(np.abs(q8["residual"])),
                           np.max(np.abs(q16["residual"])) - 1e-6)

    def test_per_element_step_shape(self):
        x = np.array([[0.01, 5.0], [-3.0, 0.2]], dtype=np.float32)
        q = quantize(x, 16, "per_element")
        self.assertEqual(np.asarray(q["step"]).shape, x.shape)

    def test_invalid_enums(self):
        x = np.ones((2,), dtype=np.float32)
        with self.assertRaises(ValueError):
            quantize(x, 12, "uniform")
        with self.assertRaises(ValueError):
            quantize(x, 8, "weird")


if __name__ == "__main__":
    unittest.main()
