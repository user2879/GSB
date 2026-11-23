import unittest

from mstc.fixedpoint import (
    FixedConfig, Gear, Rounding, argmax_tie_low, from_int, isqrt, q_div, q_mul,
    weight_to_fixed,
)


class FixedPointTests(unittest.TestCase):
    def test_isqrt_deterministic(self):
        self.assertEqual(isqrt(0), 0)
        self.assertEqual(isqrt(1), 1)
        self.assertEqual(isqrt(15), 3)
        self.assertEqual(isqrt(16), 4)
        self.assertEqual(isqrt(999999 ** 2), 999999)

    def test_gear_changes_integer_grid(self):
        cfg_fine = FixedConfig(12, Gear.FINE)
        cfg_coarse = FixedConfig(12, Gear.COARSE)
        q_fine, _, _ = from_int(5, cfg_fine)
        q_coarse, _, _ = from_int(5, cfg_coarse)
        self.assertNotEqual(q_fine, q_coarse)
        self.assertEqual(q_fine, 5 * 4096)
        self.assertEqual(q_coarse, 5 * 1024)

    def test_rounding_direction_recorded(self):
        cfg = FixedConfig(12, Gear.NORMAL)
        q, rem, direction = from_int(1, cfg)
        self.assertIn(direction, (Rounding.FLOOR, Rounding.UP, Rounding.EXACT))
        self.assertEqual(q, 1 * 2048)

    def test_division_zero_denom_is_zero(self):
        cfg = FixedConfig(12, Gear.FINE)
        q, _, d = q_div(3, 0, cfg)
        self.assertEqual((q, d), (0, Rounding.EXACT))

    def test_qmul_rescales(self):
        cfg = FixedConfig(8, Gear.FINE)
        a = weight_to_fixed(2.0, cfg)
        b = weight_to_fixed(3.0, cfg)
        q, _, _ = q_mul(a, b, cfg)
        self.assertEqual(q, weight_to_fixed(6.0, cfg))

    def test_argmax_tie_prefers_low_index(self):
        self.assertEqual(argmax_tie_low([7, 7, 3, 7]), 0)
        self.assertEqual(argmax_tie_low([1, 9, 9, 2]), 1)

    def test_determinism_repeated_calls(self):
        cfg = FixedConfig(12, Gear.NORMAL)
        first = [q_div(17, 23, cfg) for _ in range(5)]
        self.assertTrue(all(x == first[0] for x in first))


if __name__ == "__main__":
    unittest.main()
