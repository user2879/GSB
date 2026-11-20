import unittest

import numpy as np

from texmat_pipeline.noise import value_noise_field, pattern_control_points


class NoiseTests(unittest.TestCase):
    def test_deterministic_and_bounded(self):
        f1 = value_noise_field(24, 18, frequency=0.7, fingerprint=12345)
        f2 = value_noise_field(24, 18, frequency=0.7, fingerprint=12345)
        np.testing.assert_array_equal(f1, f2)
        self.assertEqual(f1.shape, (18, 24))  # row-major (H, W)
        self.assertTrue(np.all(f1 >= -1.0) and np.all(f1 <= 1.0))

    def test_fingerprint_changes_field(self):
        f1 = value_noise_field(16, 16, 1.1, fingerprint=1)
        f2 = value_noise_field(16, 16, 1.1, fingerprint=2)
        self.assertFalse(np.array_equal(f1, f2))

    def test_frequency_amplitude_effect(self):
        low = value_noise_field(32, 32, 0.3, fingerprint=7)
        high = value_noise_field(32, 32, 3.0, fingerprint=7)
        self.assertFalse(np.array_equal(low, high))

    def test_control_points_deterministic(self):
        p1 = pattern_control_points(99, 32, 32)
        p2 = pattern_control_points(99, 32, 32)
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(p1.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
