"""纯标准库 PNG 往返、定点量化与指纹稳定性的补充测试。"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhash_ledger.fixedpoint import fixed_dot, fixed_matvec, quantize_vector, to_fixed
from xhash_ledger.hashing import derive_key, fingerprint
from xhash_ledger.pngio import PngError, read_png, write_png


class PngRoundTripTests(unittest.TestCase):
    def test_write_read_roundtrip(self):
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        path = os.path.join(tempfile.mkdtemp(), "img.png")
        write_png(path, pixels)
        image = read_png(path)
        self.assertEqual((image.width, image.height), (48, 32))
        self.assertEqual(image.bit_depth, 8)
        self.assertEqual(image.color_type, 2)
        self.assertTrue(np.array_equal(image.pixels, pixels))

    def test_broken_signature_raises(self):
        path = os.path.join(tempfile.mkdtemp(), "bad.png")
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 32)
        with self.assertRaises(PngError):
            read_png(path)


class FixedPointTests(unittest.TestCase):
    def test_to_fixed_rounding(self):
        self.assertEqual(to_fixed(1.0, 8), 256)
        self.assertEqual(to_fixed(0.5, 8), 128)
        self.assertEqual(to_fixed(-0.5, 8), -128)

    def test_integer_dot_product_deterministic(self):
        a = [to_fixed(x, 8) for x in (0.25, 0.75, -0.5)]
        b = [to_fixed(x, 8) for x in (1.0, -1.0, 0.5)]
        self.assertEqual(fixed_dot(a, b), fixed_dot(list(reversed(a)), list(reversed(b))))
        matrix = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(fixed_matvec(matrix, [1, 2, 3]), [14, 32])

    def test_quantization_residual_bounded(self):
        q = quantize_vector([0.1, 0.2, 0.3, 1.999], 10)
        self.assertTrue(all(-1 <= r <= 1 for r in q.residuals))
        self.assertEqual(q.to_dict()["qstep"], "2^-10")

    def test_fingerprint_stable_and_order_independent(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(derive_key(["a", "1"]), derive_key(["a", "2"]))


if __name__ == "__main__":
    unittest.main()
