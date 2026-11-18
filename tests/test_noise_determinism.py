import unittest

import numpy as np

from texmat.noise import noise_grid, pixel_hash_grid, value_noise


class NoiseDeterminismTest(unittest.TestCase):
    def test_pixel_hash_pure_row_major(self):
        a = pixel_hash_grid(16, 8, fingerprint=7, seed=99)
        b = pixel_hash_grid(16, 8, fingerprint=7, seed=99)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (8, 16))
        self.assertEqual(a.dtype, np.uint64)

    def test_noise_bounds_and_determinism(self):
        n1 = noise_grid(32, 32, fingerprint=3, seed=5)
        n2 = noise_grid(32, 32, fingerprint=3, seed=5)
        np.testing.assert_array_equal(n1, n2)
        self.assertGreaterEqual(n1.min(), 0.0)
        self.assertLessEqual(n1.max(), 1.0)
        n3 = noise_grid(32, 32, fingerprint=4, seed=5)
        self.assertFalse(np.array_equal(n1, n3))

    def test_value_noise_deterministic_bounded(self):
        v1 = value_noise(24, 24, frequency=6, fingerprint=11, seed=13)
        v2 = value_noise(24, 24, frequency=6, fingerprint=11, seed=13)
        np.testing.assert_array_equal(v1, v2)
        self.assertEqual(v1.shape, (24, 24))
        self.assertGreaterEqual(v1.min(), 0.0)
        self.assertLessEqual(v1.max(), 1.0)

    def test_value_noise_frequency_sensitivity(self):
        low = value_noise(32, 32, frequency=2, fingerprint=1, seed=1)
        high = value_noise(32, 32, frequency=16, fingerprint=1, seed=1)
        self.assertFalse(np.array_equal(low, high))


if __name__ == "__main__":
    unittest.main()
