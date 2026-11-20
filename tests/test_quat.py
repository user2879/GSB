import unittest

import numpy as np

from animpipe.errors import E_QUAT_INVALID
from animpipe.quat import (
    align_sequence,
    angular_distance,
    nlerp,
    normalize,
    slerp,
)


class QuatTests(unittest.TestCase):
    def test_normalize_and_align_sign(self):
        q = np.array([0.0, 2.0, 0.0, 0.0])
        unit = normalize(q)
        self.assertAlmostEqual(float(np.linalg.norm(unit)), 1.0)

        seq = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 2, 1))
        seq[2, 0] = -1.0
        aligned = align_sequence(seq)
        self.assertGreaterEqual(float(np.dot(aligned[1, 0], aligned[2, 0])), 0.0)
        self.assertGreaterEqual(float(np.dot(aligned[2, 0], aligned[3, 0])), 0.0)

    def test_slerp_sign_invariant(self):
        q0 = normalize(np.array([0.9, 0.1, 0.2, 0.3]))
        q1 = normalize(np.array([0.7, 0.4, 0.5, 0.1]))
        for u in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = slerp(q0, q1, u)
            b = slerp(q0, -q1, u)
            self.assertLess(angular_distance(a, b), 1e-12)
            c = nlerp(q0, q1, u)
            d = nlerp(q0, -q1, u)
            self.assertLess(angular_distance(c, d), 1e-12)

    def test_invalid_quaternions(self):
        with self.assertRaises(E_QUAT_INVALID) as caught:
            normalize(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(caught.exception.code, "E_QUAT_INVALID")

        frames = np.ones((3, 2, 4), dtype=np.float64)
        frames[1, 0] = np.nan
        with self.assertRaises(E_QUAT_INVALID) as caught:
            align_sequence(frames)
        self.assertEqual(caught.exception.context["frame"], 1)
        self.assertEqual(caught.exception.context["joint"], 0)


if __name__ == "__main__":
    unittest.main()
