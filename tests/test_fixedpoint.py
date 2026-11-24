import math
import unittest

from fppdiag import fixedpoint as fp


class TestFixedPoint(unittest.TestCase):
    def test_half_away_signs(self):
        self.assertEqual(fp.div_half_away(5, 2), 3)
        self.assertEqual(fp.div_half_away(-5, 2), -3)
        self.assertEqual(fp.div_half_away(4, 2), 2)
        self.assertEqual(fp.div_half_away(-4, 2), -2)
        self.assertEqual(fp.div_half_away(1, 3), 0)  # 0.333 rounds down
        self.assertEqual(fp.div_half_away(2, 3), 1)  # 0.667 rounds up

    def test_saturation_envelope(self):
        self.assertEqual(fp.sat32(2**40), fp.INT32_MAX)
        self.assertEqual(fp.sat32(-(2**40)), fp.INT32_MIN)
        self.assertEqual(fp.sat32(123), 123)

    def test_rescale_roundtrip_and_null(self):
        self.assertIsNone(fp.rescale_int(None, 2, 4))
        self.assertEqual(fp.rescale_int(25, 2, 4), 2500)
        # 2507 ulps @1e-3 = 2.507; at 1e-1 that is 25.07 -> 25 (half away)
        self.assertEqual(fp.rescale_int(2507, 3, 1), 25)
        self.assertEqual(fp.rescale_int(-2507, 3, 1), -25)

    def test_mul_quant(self):
        y, sat = fp.mul_quant(25, 4, 2, 2)  # 0.25 * 0.04 = 0.01
        self.assertEqual(y, 1)
        self.assertFalse(sat)
        self.assertIsNone(fp.mul_quant(None, 4, 2, 2)[0])
        big, sat2 = fp.mul_quant(fp.INT32_MAX, fp.INT32_MAX, 0, 0)
        self.assertTrue(sat2)
        self.assertEqual(big, fp.INT32_MAX)

    def test_bps_ratio(self):
        self.assertEqual(fp.bps_ratio(1, 4), 2500)
        self.assertEqual(fp.bps_ratio(1, 3), 3333)
        self.assertEqual(fp.bps_ratio(5, 0), 0)

    def test_ilog_accuracy_all_scales(self):
        worst = 0
        for in_exp in range(7):
            for out_exp in range(7):
                for real in (0.5, 1.0, 1.5, 3.33, 10.0, 100.0, 999.99):
                    x = round(real * 10**in_exp)
                    if x <= 0:
                        continue
                    got = fp.ilog(x, in_exp, "e", out_exp)
                    ref = round(math.log(x / 10**in_exp) * 10**out_exp)
                    worst = max(worst, abs(got - ref))
        self.assertLessEqual(worst, 16)

    def test_ilog_null_and_nonpositive(self):
        self.assertIsNone(fp.ilog(None, 2, "e", 4))
        self.assertIsNone(fp.ilog(0, 2, "e", 4))
        self.assertIsNone(fp.ilog(-9, 2, "e", 4))

    def test_log_bases(self):
        self.assertAlmostEqual(fp.ilog(800, 2, "2", 6) / 1e6, 3.0, places=4)
        self.assertAlmostEqual(fp.ilog(100000, 2, "10", 6) / 1e6, 3.0, places=3)


if __name__ == "__main__":
    unittest.main()
