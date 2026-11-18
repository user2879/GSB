from __future__ import annotations

import unittest

import numpy as np

from renderlab.engine import derive_initial_latent, encode_prompt_bag
from renderlab.png import decode_rgb_png, encode_rgb_png


class DeterminismTests(unittest.TestCase):
    def test_prompt_bag_is_order_independent_and_stable(self) -> None:
        vocab = ("amber", "arch", "cloud", "forest")
        first = encode_prompt_bag(vocab, 8, ("amber", "cloud", "arch"))
        second = encode_prompt_bag(vocab, 8, ("arch", "amber", "cloud"))
        third = encode_prompt_bag(vocab, 8, ("arch", "amber", "cloud"))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(second, third)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=12)

    def test_initial_latent_is_seed_prompt_and_tier_deterministic(self) -> None:
        from renderlab.config import ResolutionTier

        tier = ResolutionTier("tier", 8, 8)
        first = derive_initial_latent(seed=11, prompt_fingerprint="abc", tier=tier)
        second = derive_initial_latent(seed=11, prompt_fingerprint="abc", tier=tier)
        third = derive_initial_latent(seed=13, prompt_fingerprint="abc", tier=tier)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(first.shape, (8, 8, 4))

    def test_png_roundtrip_and_byte_determinism(self) -> None:
        pixels = np.asarray(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[254, 255, 128], [7, 8, 9]],
            ],
            dtype=np.uint8,
        )
        first = encode_rgb_png(pixels)
        second = encode_rgb_png(pixels.copy())
        self.assertEqual(first, second)
        np.testing.assert_array_equal(decode_rgb_png(first), pixels)


if __name__ == "__main__":
    unittest.main()
