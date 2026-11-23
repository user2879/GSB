import math
import unittest

import numpy as np

from hotblock_diag.fixedpoint import (deterministic_round_float, quantize,
                                      score_block_pair, signed_range)
from tests._harness import make_fixedpoint, make_prefilter


class FixedPointTests(unittest.TestCase):
    def test_quantize_uses_declared_frac_bits(self):
        values = np.array([0.5, -0.5, 0.25, 1.0], dtype=np.float32)
        q = quantize(values, frac_bits=4, rounding="nearest_even")
        self.assertEqual(q.tolist(), [7, -8, 4, 7])

    def test_signed_range_and_saturation(self):
        lo, hi = signed_range(8)
        self.assertEqual((lo, hi), (-128, 127))
        cfg = make_fixedpoint(acc_bits=8, frac_bits=4, quant_gear="q4")
        q_block = np.full((2, 3), 1.0, dtype=np.float32)
        k_block = np.full((2, 3), 1.0, dtype=np.float32)
        result = score_block_pair(
            0, 0, q_block, k_block, cfg.frac_bits, cfg.acc_bits,
            cfg.saturation, cfg.rounding, temperature=1.0,
        )
        self.assertTrue(result.overflows)
        self.assertLessEqual(max(max(row) for row in result.accumulators), 127)

    def test_determinism_two_runs_identical(self):
        rng = np.random.default_rng(11)
        q = rng.normal(size=(4, 8)).astype(np.float32)
        k = rng.normal(size=(4, 8)).astype(np.float32)
        cfg = make_fixedpoint()
        first = score_block_pair(1, 2, q, k, cfg.frac_bits, cfg.acc_bits,
                                 cfg.saturation, cfg.rounding, 1.25)
        second = score_block_pair(1, 2, q, k, cfg.frac_bits, cfg.acc_bits,
                                  cfg.saturation, cfg.rounding, 1.25)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_rounding_modes_are_deterministic(self):
        value = 2.5
        self.assertEqual(deterministic_round_float(value, "truncate"), 2.0)
        self.assertEqual(deterministic_round_float(value, "nearest_even"), 2.0)
        self.assertEqual(deterministic_round_float(value, "nearest_up"), 3.0)
        self.assertEqual(deterministic_round_float(value, "floor"), 2.0)
        self.assertEqual(deterministic_round_float(-2.5, "nearest_even"), -2.0)
        self.assertEqual(deterministic_round_float(-2.5, "floor"), -3.0)

    def test_scale_uses_temperature(self):
        q = np.ones((1, 2), dtype=np.float32)
        k = np.ones((1, 2), dtype=np.float32)
        cfg = make_fixedpoint()
        result = score_block_pair(0, 0, q, k, cfg.frac_bits, cfg.acc_bits,
                                  cfg.saturation, cfg.rounding,
                                  temperature=4.0)
        expected_scale = 1.0 / (256.0 * 256.0 * 4.0)
        self.assertTrue(math.isclose(result.scale_factor, expected_scale))
        self.assertEqual(len(result.fingerprint), 16)


if __name__ == "__main__":
    unittest.main()
