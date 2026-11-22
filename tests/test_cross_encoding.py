from __future__ import annotations

import unittest

from feature_pipeline.cross import encode_combination


class CrossEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = {
            "a": ["x", "y", "x", "z", "y"],
            "b": ["m", "m", "n", "n", "m"],
        }

    def test_dense_hash_encoding_is_repeatable(self) -> None:
        settings = {
            "cardinality_budget": 16,
            "overflow_strategy": "hash_bucket",
            "hash_buckets": 8,
            "hash_seed": 11,
        }
        first = encode_combination(("a", "b"), self.values, crossing=settings)
        second = encode_combination(("a", "b"), self.values, crossing=settings)
        self.assertEqual(first, second)
        self.assertEqual(first["content"]["encoding_mode"], "dense_hash")
        self.assertEqual(first["content"]["unique_count"], 4)
        self.assertEqual(first["content"]["encoded_cardinality"], 4)
        self.assertIsNone(first["degradation"])

    def test_hash_bucket_degradation_and_skip(self) -> None:
        bucket_settings = {
            "cardinality_budget": 2,
            "overflow_strategy": "hash_bucket",
            "hash_buckets": 9,
            "hash_seed": 11,
        }
        bucketed = encode_combination(("a", "b"), self.values, crossing=bucket_settings)
        self.assertEqual(bucketed["content"]["encoding_mode"], "hash_bucket")
        self.assertLessEqual(bucketed["content"]["encoded_cardinality"], 2)
        self.assertEqual(bucketed["degradation"]["hash_buckets"], 2)
        self.assertEqual(bucketed["degradation"]["action"], "hash_bucket")

        skip_settings = {**bucket_settings, "overflow_strategy": "skip"}
        skipped = encode_combination(("a", "b"), self.values, crossing=skip_settings)
        self.assertEqual(skipped["content"]["encoding_mode"], "skipped")
        self.assertEqual(skipped["content"]["encoded"], [])
        self.assertEqual(skipped["degradation"]["action"], "skip")


if __name__ == "__main__":
    unittest.main()
