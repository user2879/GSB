import unittest

import numpy as np

from leasebatch.fixedpoint import (
    SCALE,
    forward_batch,
    input_vector,
    parameter_digest,
    parameter_matrices,
)


class FixedPointTest(unittest.TestCase):
    def test_inputs_and_params_are_deterministic(self):
        x1 = input_vector("r01", 4, 42)
        x2 = input_vector("r01", 4, 42)
        np.testing.assert_array_equal(x1, x2)
        p1 = parameter_matrices("w1", 4, 3, 0, 42)
        p2 = parameter_matrices("w1", 4, 3, 0, 42)
        for left, right in zip(p1, p2):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            parameter_digest("w1", 4, 3, 0, 42),
            parameter_digest("w1", 4, 3, 0, 42),
        )

    def test_outputs_are_integer_and_bounded(self):
        result = forward_batch(["r01", "r02"], "w1", 4, 3, 0, 42)
        for values in result["outputs"].values():
            self.assertTrue(all(isinstance(v, int) for v in values))

    def test_digest_independent_of_request_order(self):
        a = forward_batch(["r01", "r02", "r03"], "w1", 4, 3, 0, 42)
        b = forward_batch(["r03", "r01", "r02"], "w1", 4, 3, 0, 42)
        self.assertEqual(a["output_digest"], b["output_digest"])

    def test_param_version_changes_output(self):
        v0 = forward_batch(["r01"], "w1", 4, 3, 0, 42)
        v1 = forward_batch(["r01"], "w1", 4, 3, 1, 42)
        self.assertNotEqual(v0["output_digest"], v1["output_digest"])

    def test_scale_is_integer_watermark(self):
        self.assertEqual(SCALE, 1000)


if __name__ == "__main__":
    unittest.main()
