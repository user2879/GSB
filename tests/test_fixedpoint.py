import math
import unittest

from mm_pipeline.fixedpoint import (
    scale,
    from_float,
    to_float,
    fmul,
    fdiv_scaled,
    fdiv_round,
    exp_fp,
    softmax_two,
    clamp_confidence,
    quant_shift,
    QUANT_LEVELS,
)


class FixedPointTests(unittest.TestCase):
    def test_quant_shift_enum(self):
        self.assertEqual([quant_shift(q) for q in QUANT_LEVELS], [2, 3, 4])

    def test_round_trip_and_half_up(self):
        for fb in (8, 16, 24):
            s = scale(fb)
            self.assertEqual(from_float(0.5, fb), s // 2)
            self.assertEqual(from_float(-0.5, fb), -(s // 2))
            tolerance = 2.0 / scale(fb)
            self.assertLess(abs(to_float(from_float(0.3, fb), fb) - 0.3), tolerance)

    def test_mul_div_determinism(self):
        fb = 16
        s = scale(fb)
        quarter = s // 4
        self.assertEqual(fmul(quarter, quarter, fb), s // 16)
        self.assertEqual(fdiv_scaled(s, from_float(2.0, fb), fb), s // 2)
        self.assertEqual(fdiv_round(3, 2, fb), int(1.5 * s))

    def test_exp_accuracy(self):
        fb = 16
        for value in (-2.0, -0.5, 0.0, 0.7, 1.5):
            got = to_float(exp_fp(from_float(value, fb), fb), fb)
            self.assertLess(abs(got - math.exp(value)), 1e-3)

    def test_softmax_stable_and_exact_sum(self):
        fb = 16
        s = scale(fb)
        wa, wb = softmax_two(s, s, fb, s)
        self.assertEqual((wa, wb), (s // 2, s // 2))
        wa, wb = softmax_two(from_float(2, fb), from_float(0, fb), fb, s)
        self.assertEqual(wa + wb, s)
        self.assertAlmostEqual(to_float(wa, fb), 0.8808, places=3)

    def test_clamp(self):
        fb = 16
        s = scale(fb)
        self.assertEqual(clamp_confidence(2 * s, fb), s)
        self.assertEqual(clamp_confidence(-5, fb), 0)


if __name__ == "__main__":
    unittest.main()
