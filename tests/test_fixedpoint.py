import unittest

import numpy as np

from texmat_pipeline.fixedpoint import (
    FixedSpec,
    quantize_float,
    round_div_int,
    weighted_blend,
    fixed_to_uint8,
)


class FixedPointTests(unittest.TestCase):
    def setUp(self):
        self.unit = FixedSpec(frac_bits=8, levels=256, lo=0.0, hi=1.0)
        self.signed = FixedSpec(frac_bits=8, levels=512, lo=-1.0, hi=1.0)

    def test_quantize_endpoints_and_step(self):
        vals = np.array([0.0, 1.0], dtype=np.float64)
        q, boundary = quantize_float(vals, self.unit, "half_even")
        self.assertEqual(q.tolist(), [0, 256])
        self.assertTrue(boundary.all())
        self.assertEqual(self.unit.step, 1)

    def test_half_up_vs_half_even_tie_decisions(self):
        # Construct an exact half-step tie with a coarse quantization grid.
        spec = FixedSpec(frac_bits=4, levels=8, lo=0.0, hi=1.0)
        # step = round(0.125 * 16) = 2; value 0.0625 -> index 0.5 tie
        vals = np.array([0.0625])
        q_up, b_up = quantize_float(vals, spec, "half_up")
        q_even, b_even = quantize_float(vals, spec, "half_even")
        self.assertEqual(int(q_up[0]), 2)
        self.assertEqual(int(q_even[0]), 0)
        self.assertTrue(bool(b_up[0]) and bool(b_even[0]))

    def test_truncate_and_floor_are_deterministic(self):
        vals = np.linspace(-1.0, 1.0, 33)
        q1, _ = quantize_float(vals, self.signed, "truncate")
        q2, _ = quantize_float(vals, self.signed, "truncate")
        np.testing.assert_array_equal(q1, q2)
        qf, _ = quantize_float(vals, self.signed, "floor")
        # floor is always <= truncate for negative non-integers
        self.assertTrue(np.all(qf <= q1))

    def test_round_div_tie_mask(self):
        a = np.array([3, -3, 5, -5, 4], dtype=np.int64)
        q, tie = round_div_int(a, 2, "half_even")
        self.assertEqual(q.tolist(), [2, -2, 2, -2, 2])
        # 3/2 and 5/2 both leave remainder 1 = half of 2; 4/2 is exact
        self.assertEqual(tie.tolist(), [True, True, True, True, False])
        qu, _ = round_div_int(a, 2, "half_up")
        self.assertEqual(qu.tolist(), [2, -2, 3, -3, 2])

    def test_weighted_blend_modes_and_positive_weight_rule(self):
        a = np.array([100, 200], dtype=np.int64)
        b = np.array([300, 400], dtype=np.int64)
        out_norm, tie = weighted_blend([(a, 1), (b, 1)], self.unit, "half_even", True)
        # 300/400 exceed the unit domain and are clamped to 256
        self.assertEqual(out_norm.tolist(), [200, 256])
        out_sum, _ = weighted_blend([(a, 1), (b, 1)], self.unit, "half_even", False)
        self.assertEqual(out_sum.tolist(), [256, 256])  # clamped to unit max
        with self.assertRaises(ValueError):
            weighted_blend([(a, 0), (b, 0)], self.unit, "half_even", True)

    def test_uint8_preview_maps_full_domain(self):
        q, _ = quantize_float(np.array([0.0, 0.5, 1.0]), self.unit, "half_even")
        preview = fixed_to_uint8(q, self.unit)
        self.assertEqual(preview.tolist(), [0, 128, 255])


if __name__ == "__main__":
    unittest.main()
