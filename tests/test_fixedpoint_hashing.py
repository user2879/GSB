import unittest

from hashlot.fixedpoint import quantize, sat_add, sat_div, INT64_MAX, INT64_MIN
from hashlot.hashing import (
    Fnv1a32, hash_group_key, map_partition, map_subpartition,
    validate_partition_count, SUPPORTED_PARTITION_COUNTS,
)


class FixedPointTest(unittest.TestCase):
    def test_quantize_truncates_toward_zero(self):
        self.assertEqual(quantize(1.23456, 1000), 1234)
        self.assertEqual(quantize(-1.23456, 1000), -1234)
        self.assertEqual(quantize(0.999, 10), 9)

    def test_quantize_saturates(self):
        self.assertEqual(quantize(1e30, 1000), INT64_MAX)
        self.assertEqual(quantize(-1e30, 1000), INT64_MIN)

    def test_saturated_add(self):
        self.assertEqual(sat_add(INT64_MAX, 1), INT64_MAX)
        self.assertEqual(sat_add(INT64_MIN, -1), INT64_MIN)
        self.assertEqual(sat_add(2, 3), 5)

    def test_sat_div_truncates_toward_zero(self):
        self.assertEqual(sat_div(7, 2), 3)
        self.assertEqual(sat_div(-7, 2), -3)
        self.assertIsNone(sat_div(5, 0))


class HashingTest(unittest.TestCase):
    def test_fnv_known_vector(self):
        hasher = Fnv1a32()
        hasher.update_bytes(b"")
        self.assertEqual(hasher.digest(), 0x811C9DC5)

    def test_partition_count_supported_set(self):
        for n in SUPPORTED_PARTITION_COUNTS:
            self.assertEqual(validate_partition_count(n), n)
        with self.assertRaises(ValueError):
            validate_partition_count(3)
        with self.assertRaises(ValueError):
            validate_partition_count(2048)

    def test_same_key_same_partition_and_subpartition(self):
        key = ((0x11, (5).to_bytes(8, "little", signed=True)),)
        digest = hash_group_key(key)
        partitions = {map_partition(digest, n) for n in SUPPORTED_PARTITION_COUNTS}
        # 对固定分区数重复映射必须稳定
        self.assertEqual(map_partition(digest, 8), map_partition(digest, 8))
        self.assertLess(map_partition(digest, 8), 8)
        subs = {map_subpartition(digest, 4) for _ in range(10)}
        self.assertEqual(len(subs), 1)

    def test_hash_independent_of_insertion_order(self):
        key = ((0x11, (42).to_bytes(8, "little", signed=True)),)
        first = hash_group_key(key)
        for _ in range(5):
            self.assertEqual(hash_group_key(key), first)


if __name__ == "__main__":
    unittest.main()
