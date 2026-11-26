import unittest

from fixedpoint_render.fixedpoint import (
    q_div_round, q_mul, q_scale, parse_decimal_q, quantize, q_to_uint8_clamp,
)


class FixedPointTests(unittest.TestCase):
    def test_parse_decimal_exact(self):
        self.assertEqual(parse_decimal_q("0.25", 8), 64)
        self.assertEqual(parse_decimal_q("1.0", 8), 256)
        self.assertEqual(parse_decimal_q("-0.5", 8), -128)
        self.assertEqual(parse_decimal_q("3", 12), 3 * 4096)

    def test_rounding_modes_differ(self):
        # 3/2 division distinguishes trunc/floor/nearest on negatives.
        self.assertEqual(q_div_round(-3, 2, "trunc"), -1)
        self.assertEqual(q_div_round(-3, 2, "floor"), -2)
        self.assertEqual(q_div_round(-3, 2, "nearest"), -2)
        self.assertEqual(q_div_round(3, 2, "nearest"), 2)

    def test_mul_records_residual(self):
        scale = q_scale(8)
        value, trace = q_mul(scale // 2, scale // 4, 8, "trunc")  # 0.5*0.25
        self.assertEqual(value, scale // 8)
        self.assertEqual(trace.residual, 0)

    def test_quantize_changes_integer_and_residual(self):
        # Q16 value with a non-multiple-of-256 low part forces a residual.
        wide = parse_decimal_q("1.0", 16) + 100
        narrow, trace = quantize(wide, 16, 8, "trunc")
        self.assertNotEqual(wide, narrow)
        self.assertEqual(trace.raw, wide)
        self.assertEqual(trace.residual, 100)

    def test_qmode_changes_quantized_result(self):
        # Low 8 bits = 200 > 128: trunc rounds down, nearest rounds up.
        wide = parse_decimal_q("1.0", 16) + 200
        trunc, _ = quantize(wide, 16, 8, "trunc")
        near, _ = quantize(wide, 16, 8, "nearest")
        self.assertNotEqual(trunc, near)

    def test_uint8_clamp(self):
        scale = q_scale(8)
        pixel, _ = q_to_uint8_clamp(300 * scale, 8, "nearest")
        self.assertEqual(pixel, 255)
        pixel, _ = q_to_uint8_clamp(-5 * scale, 8, "nearest")
        self.assertEqual(pixel, 0)


if __name__ == "__main__":
    unittest.main()
