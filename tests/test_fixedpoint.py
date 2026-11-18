import unittest

import numpy as np

from texmat.fixedpoint import (qmul, quantize_float, rescale_array,
                               rounded_div, saturate_i32, to_uint8)


class FixedPointTest(unittest.TestCase):
    def test_rounded_div_modes(self):
        self.assertEqual(rounded_div(5, 2, "truncate"), 2)
        self.assertEqual(rounded_div(-5, 2, "truncate"), -2)
        self.assertEqual(rounded_div(5, 2, "half_up"), 3)
        self.assertEqual(rounded_div(-5, 2, "half_up"), -3)
        self.assertEqual(rounded_div(5, 2, "half_even"), 2)
        self.assertEqual(rounded_div(7, 2, "half_even"), 4)

    def test_rescale_widen_then_narrow_identity(self):
        values = np.array([-100, -1, 0, 1, 99, 12345], dtype=np.int64)
        widened = rescale_array(values, 8, 12, "truncate")
        back = rescale_array(widened, 12, 8, "truncate")
        np.testing.assert_array_equal(back, values)

    def test_rescale_rounding_modes_at_edge(self):
        values = np.array([3, 5, 7], dtype=np.int64)
        np.testing.assert_array_equal(rescale_array(values, 2, 1, "truncate"), [1, 2, 3])
        np.testing.assert_array_equal(rescale_array(values, 2, 1, "half_up"), [2, 3, 4])
        np.testing.assert_array_equal(rescale_array(values, 2, 1, "half_even"), [2, 2, 4])

    def test_qmul(self):
        a = np.array([128], dtype=np.int64)
        np.testing.assert_array_equal(qmul(a, a, 8, "half_even"), [64])
        one = np.array([256], dtype=np.int64)
        x = np.array([-123, 0, 200], dtype=np.int64)
        np.testing.assert_array_equal(qmul(one, x, 8, "half_even"), x)

    def test_quantize_float(self):
        self.assertEqual(quantize_float(0.5, 8, "truncate"), 128)
        self.assertEqual(quantize_float(0.00390625 * 2.5, 8, "half_even"), 2)  # 2.5 -> even 2

    def test_saturate_and_bytes(self):
        values = np.array([-50, 0, 200, 300], dtype=np.int64)
        self.assertTrue(np.all(saturate_i32(values) >= -(2**31)))
        bytes_out = to_uint8(np.array([-5, 0, 255, 300], dtype=np.int64), 8, "half_even")
        np.testing.assert_array_equal(bytes_out, [0, 0, 255, 255])


if __name__ == "__main__":
    unittest.main()
