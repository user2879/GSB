import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.hashing import (
    array_fingerprint, cache_key, derive_seed, fingerprint, random_state,
)
from render_workbench.png_codec import decode_png, encode_png


class HashingTests(unittest.TestCase):
    def test_canonical_fingerprint_key_order_independent(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}),
                         fingerprint({"b": 2, "a": 1}))

    def test_derived_rng_isolated_and_deterministic(self):
        a = random_state("x", 1).standard_normal(5)
        b = random_state("x", 1).standard_normal(5)
        c = random_state("x", 2).standard_normal(5)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_array_fingerprint_detects_dtype_and_shape(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        same = arr.copy()
        diff = arr.astype(np.float32)
        self.assertEqual(array_fingerprint(arr), array_fingerprint(same))
        self.assertNotEqual(array_fingerprint(arr), array_fingerprint(diff))

    def test_cache_key_uses_upstream_fingerprints(self):
        k1 = cache_key("denoise", {"s": 0}, ["fp-a"])
        k2 = cache_key("denoise", {"s": 0}, ["fp-b"])
        k3 = cache_key("denoise", {"s": 1}, ["fp-a"])
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(k1, cache_key("denoise", {"s": 0}, ["fp-a"]))

    def test_seed_in_uint32(self):
        seed = derive_seed("anything", {"k": "v"})
        self.assertTrue(0 <= seed < 2 ** 32)


class PngTests(unittest.TestCase):
    def test_roundtrip_deterministic(self):
        rng = np.random.RandomState(0)
        pixels = rng.randint(0, 256, size=(7, 11, 3)).astype(np.uint8)
        png_a = encode_png(pixels)
        png_b = encode_png(pixels)
        self.assertEqual(png_a, png_b)
        decoded = decode_png(png_a)
        np.testing.assert_array_equal(decoded, pixels)

    def test_clipped_values_roundtrip(self):
        pixels = np.zeros((3, 3, 3), dtype=np.uint8)
        pixels[0, 0] = [0, 128, 255]
        np.testing.assert_array_equal(decode_png(encode_png(pixels)), pixels)

    def test_rejects_foreign_png(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"not a png")
            path = handle.name
        try:
            with open(path, "rb") as image:
                with self.assertRaises(ValueError):
                    decode_png(image.read())
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
