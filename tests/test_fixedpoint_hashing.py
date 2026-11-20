import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import fixedpoint as fp
from dhj.hashing import fnv1a64, hash_int, key_partition, row_digest
from dhj.jsonio import dumps_canonical


class FixedPointTests(unittest.TestCase):
    def test_round_trip_integers(self):
        for bits in (0, 4, 8, 16):
            for value in (-17, -1, 0, 1, 17, 100000):
                q = fp.quantize(value, bits)
                self.assertEqual(q, value * fp.scale_factor(bits))
                self.assertEqual(fp.dequantize(q, bits), float(value))

    def test_half_away_from_zero(self):
        q = fp.quantize(0.5, 0)
        self.assertEqual(q, 1)
        self.assertEqual(fp.quantize(-0.5, 0), -1)
        self.assertEqual(fp.quantize(0.125, 2), 1)

    def test_null_and_bounds(self):
        self.assertEqual(fp.quantize(None, 4), fp.NULL_Q)
        self.assertIsNone(fp.dequantize(fp.NULL_Q, 4))
        with self.assertRaises(ValueError):
            fp.quantize(2**62, 4)

    def test_error_bound(self):
        self.assertEqual(fp.error_bound(4), 1 / 32)
        self.assertEqual(fp.sum_error_bound(10, 4), 10 / 32)
        self.assertLessEqual(
            abs(fp.quantization_error(0.12, 4)), fp.error_bound(4)
        )


class HashingTests(unittest.TestCase):
    def test_known_fnv_vector(self):
        self.assertEqual(
            fnv1a64(b""), 0xCBF29CE484222325
        )
        self.assertEqual(fnv1a64(b"a"), 0xAF63DC4C8601EC8C)

    def test_deterministic_partition(self):
        q = fp.quantize(11, 4)
        p1 = key_partition(q, 7)
        p2 = key_partition(q, 7)
        self.assertEqual(p1, p2)
        self.assertIn(p1, range(7))

    def test_partition_covers_range(self):
        seen = {key_partition(hash_int(i), 13) for i in range(200)}
        self.assertTrue(seen)
        self.assertTrue(all(0 <= p < 13 for p in seen))

    def test_row_digest_stable_and_distinct(self):
        a = row_digest("t", 1, "left", "k", {"k": 1}, 1)
        b = row_digest("t", 1, "left", "k", {"k": 1}, 1)
        c = row_digest("t", 2, "left", "k", {"k": 1}, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_canonical_bytes_deterministic(self):
        payload = {"b": 1, "a": [1, 2, {"c": 3}]}
        self.assertEqual(dumps_canonical(payload), dumps_canonical(payload))


if __name__ == "__main__":
    unittest.main()
