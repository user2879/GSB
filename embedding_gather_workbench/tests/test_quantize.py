import unittest

import numpy as np

from egw.quantize import (
    accumulator_bounds,
    divide_mean,
    quantize_row,
    quantize_step,
    saturating_add,
)


class QuantizeTests(unittest.TestCase):
    def test_step_and_truncation(self):
        vector = np.array([0.123, -0.123, 1.0, -2.5])
        ints, step, residual = quantize_row(vector, 16, 8)
        self.assertEqual(step, 1.0 / 256.0)
        np.testing.assert_array_equal(ints, np.array([31, -31, 256, -640], dtype=np.int64))
        np.testing.assert_allclose(residual, vector - ints * step, rtol=0, atol=1e-18)

    def test_clipping_by_gear(self):
        vector = np.array([100000.0, -100000.0])
        ints_16, _, _ = quantize_row(vector, 16, 8)
        ints_12, _, _ = quantize_row(vector, 12, 4)
        self.assertEqual(ints_16[0], 32767)
        self.assertEqual(ints_16[1], -32768)
        self.assertEqual(ints_12[0], 2047)
        self.assertNotEqual(ints_16.tolist(), ints_12.tolist())

    def test_determinism_across_runs(self):
        rng = np.random.default_rng(7)
        vector = rng.normal(size=16)
        first = quantize_row(vector, 16, 8)
        second = quantize_row(vector, 16, 8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])

    def test_bit_width_changes_representation(self):
        vector = np.array([0.123456])
        ints_8, _, _ = quantize_row(vector, 16, 8)
        ints_4, _, _ = quantize_row(vector, 16, 4)
        self.assertNotEqual(ints_8[0], ints_4[0])
        self.assertEqual(quantize_step(4), 1.0 / 16.0)

    def test_saturating_add(self):
        amin, amax = accumulator_bounds(8, 2)
        acc = np.array([amax - 2], dtype=np.int64)
        out = saturating_add(acc, np.array([10], dtype=np.int64), amin, amax)
        self.assertEqual(out[0], amax)
        out = saturating_add(np.array([amin + 2]), np.array([-10]), amin, amax)
        self.assertEqual(out[0], amin)

    def test_mean_division_rounding_modes(self):
        total = np.array([5, -5, 4, -4], dtype=np.int64)
        q_trunc, r_trunc = divide_mean(total, 2, "TRUNCATE")
        np.testing.assert_array_equal(q_trunc, [2, -2, 2, -2])
        np.testing.assert_array_equal(r_trunc, total - q_trunc * 2)
        q_half, r_half = divide_mean(total, 3, "HALF_UP")
        np.testing.assert_array_equal(q_half, [2, -2, 1, -1])
        np.testing.assert_array_equal(r_half, total - q_half * 3)
        q_even, _ = divide_mean(np.array([3, 5], dtype=np.int64), 2, "HALF_EVEN")
        np.testing.assert_array_equal(q_even, [2, 2])
        q_floor, _ = divide_mean(total, 3, "FLOOR")
        np.testing.assert_array_equal(q_floor, [1, -2, 1, -2])


if __name__ == "__main__":
    unittest.main()
