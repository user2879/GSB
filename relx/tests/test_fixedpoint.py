import unittest

from relx.errors import E_CONFIG_INVALID, RelxError
from relx.fixedpoint import (
    ALLOWED_Q_LEVELS,
    ALLOWED_WIDTHS,
    half_up_divide,
    quantize,
    saturate,
)


class FixedPointTest(unittest.TestCase):
    def test_quantize_round_half_up(self):
        self.assertEqual(quantize(0.42, 16, 4), 4200)
        self.assertEqual(quantize(0.12345, 16, 4), 1235)
        self.assertEqual(quantize(-0.12345, 16, 4), -1235)
        self.assertEqual(quantize(1.572, 16, 4), 15720)

    def test_saturation_widths(self):
        self.assertEqual(saturate(999999, 8), 127)
        self.assertEqual(saturate(-999999, 8), -128)
        self.assertEqual(saturate(999999, 16), 32767)
        self.assertEqual(saturate(999999, 32), 999999)

    def test_quant_levels_change_integer(self):
        values = {q: quantize(0.42, 32, q) for q in ALLOWED_Q_LEVELS}
        self.assertEqual(len(set(values.values())), 3)
        self.assertEqual(values[2], 42)
        self.assertEqual(values[8], 42000000)

    def test_invalid_width_and_quant(self):
        with self.assertRaises(RelxError) as ctx:
            quantize(0.1, 7, 4)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)
        with self.assertRaises(RelxError):
            saturate(1, 12)

    def test_half_up_divide_signed(self):
        self.assertEqual(half_up_divide(5, 2), 3)
        self.assertEqual(half_up_divide(-5, 2), -3)
        self.assertEqual(half_up_divide(4, 2), 2)
        with self.assertRaises(ValueError):
            half_up_divide(1, 0)


if __name__ == "__main__":
    unittest.main()
