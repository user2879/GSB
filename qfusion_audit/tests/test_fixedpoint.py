import unittest

from qfusion.fixedpoint import (clip_int, quantize, quantization_step,
                                round_half_up, shift_right_half_up,
                                stable_key)


class FixedPointTest(unittest.TestCase):
    def test_round_half_up_positive_and_negative(self):
        self.assertEqual(round_half_up(2.5, 1.0), 3)
        self.assertEqual(round_half_up(2.4, 1.0), 2)
        self.assertEqual(round_half_up(-2.5, 1.0), -3)
        self.assertEqual(round_half_up(-2.4, 1.0), -2)

    def test_quantize_step_grows_with_gear(self):
        fine = quantize(0.37, 8, 1)
        coarse = quantize(0.37, 8, 4)
        self.assertEqual(fine.step, quantization_step(8, 1))
        self.assertEqual(coarse.step, 4 * fine.step)
        # 切换档位真实改变整数结果或残差（要求 2）。
        self.assertTrue(fine.q != coarse.q or fine.residual != coarse.residual)

    def test_residual_within_half_step_and_direction(self):
        q = quantize(0.37, 8, 1)
        self.assertLessEqual(abs(q.residual), q.step / 2 + 1e-15)
        self.assertIn(q.direction, ("up", "down", "exact"))
        exact = quantize(0.5, 1, 1)
        self.assertEqual(exact.q, 1)
        self.assertEqual(exact.direction, "exact")
        self.assertEqual(exact.residual, 0.0)

    def test_determinism_repeat(self):
        a = quantize(0.123456, 10, 2).to_dict()
        b = quantize(0.123456, 10, 2).to_dict()
        self.assertEqual(a, b)

    def test_shift_right_half_up_negative_aware(self):
        self.assertEqual(shift_right_half_up(7, 2), 2)
        self.assertEqual(shift_right_half_up(-7, 2), -2)
        self.assertEqual(shift_right_half_up(8, 0), 8)

    def test_clip(self):
        self.assertEqual(clip_int(5, 0, 4), 4)
        self.assertEqual(clip_int(-5, 0, 4), 0)
        self.assertEqual(clip_int(2, 0, 4), 2)

    def test_stable_key_ordering(self):
        k1 = stable_key("r1", "i1", "fusion")
        k2 = stable_key("r1", "i2", "fusion")
        self.assertLess(k1, k2)


if __name__ == "__main__":
    unittest.main()
