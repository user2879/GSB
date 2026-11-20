import unittest

import numpy as np

from gradcomp.errors import E_CONFIG, GradCompError
from gradcomp.quantize import (Quantizer, combined_elements_bound,
                               vector_relative_bound)


class TestQuantizer(unittest.TestCase):
    def test_floor_error_sign_and_bound(self):
        q = Quantizer(2, "floor")  # s=100, step=0.01
        x = np.array([0.12345, -0.12345, 1.0, 0.0])
        rt = q.roundtrip(x)
        np.testing.assert_array_less(rt["errors"], 1e-12)  # floor: e <= 0
        self.assertTrue(rt["bound_respected"])
        self.assertLessEqual(rt["measured_max_abs_error"],
                             rt["single_value_bound"] * (1 + 1e-9) + 1e-12)

    def test_nearest_half_bound(self):
        q = Quantizer(2, "nearest")
        x = np.array([0.12345, 0.12399])
        rt = q.roundtrip(x)
        self.assertAlmostEqual(rt["single_value_bound"], 0.005)
        self.assertTrue(rt["bound_respected"])

    def test_toward_zero_bound(self):
        q = Quantizer(3, "toward_zero")
        x = np.array([0.001234, -0.001234])
        rt = q.roundtrip(x)
        self.assertTrue(rt["bound_respected"])
        self.assertAlmostEqual(rt["single_value_bound"], 0.001)

    def test_integer_exact_values(self):
        q = Quantizer(2, "floor")
        x = np.array([0.25, -0.75])
        rt = q.roundtrip(x)
        np.testing.assert_array_equal(rt["errors"], 0.0)

    def test_vector_and_combined_bounds(self):
        vec = vector_relative_bound(100.0, 64, np.linalg.norm(np.ones(64)),
                                    "floor")
        self.assertAlmostEqual(vec["vector_abs_bound_2norm"],
                               0.01 * np.sqrt(64))
        comb = combined_elements_bound(100.0, 4, "floor")
        self.assertAlmostEqual(comb["combined_abs_bound_l1"], 0.04)
        self.assertAlmostEqual(comb["combined_abs_bound_2norm"],
                               0.01 * 2.0)

    def test_unknown_truncation_rejected(self):
        with self.assertRaises(GradCompError) as ctx:
            Quantizer(2, "bogus").quantize(np.zeros(3))
        self.assertEqual(ctx.exception.code, E_CONFIG)


if __name__ == "__main__":
    unittest.main()
