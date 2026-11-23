from __future__ import annotations

import unittest

from near_dup.config import validate_config_payload
from near_dup.features import normalize_and_window
from near_dup.fixedpoint import similarity_units
from near_dup.hashing import compute_minhash, pair_similarity, signed_projection_bucket
from near_dup.sample_data import BASE_CONFIG, VOCABULARY
from near_dup.text_store import TextRecord


class DeterminismTests(unittest.TestCase):
    def test_normalization_window_and_padding_are_repeatable(self) -> None:
        raw = dict(BASE_CONFIG)
        raw["vocabulary"] = VOCABULARY + ["alpha"]
        raw["window_count"] = 4
        config, failures = validate_config_payload(raw)
        self.assertEqual(failures, [])
        record = TextRecord(
            "N001",
            ("ＡＬＰＨＡ", "甲", "甲", "甲", "甲", "甲", "乙", "乙", "乙"),
            "forum",
            1,
        )
        first = normalize_and_window(record, config)
        second = normalize_and_window(record, config)
        self.assertEqual(first.payload(), second.payload())
        self.assertEqual(first.normalized_terms[0], "alpha")
        self.assertEqual(len(first.windows), 4)
        self.assertEqual(len(first.padding_positions), 3)
        self.assertTrue(all(len(window) == first.window_size for window in first.windows))
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_signature_length_layout_and_bucket_width(self) -> None:
        config, _ = validate_config_payload(BASE_CONFIG)
        record = TextRecord("N002", tuple(VOCABULARY[:12]), "wiki", 1)
        result = compute_minhash(normalize_and_window(record, config), config)
        self.assertEqual(len(result.signature), config.signature_dimension)
        self.assertEqual(len({lane for _, lane in result.layout}), config.signature_dimension)
        self.assertEqual({window for window, _ in result.layout}, {0, 1, 2, 3})
        bucket = signed_projection_bucket(result.signature, config)
        self.assertEqual(len(bucket), config.bucket_bits)
        self.assertEqual(bucket, signed_projection_bucket(result.signature, config))

    def test_fixed_point_exact_threshold_is_boundary(self) -> None:
        config, _ = validate_config_payload(BASE_CONFIG)
        signature = tuple(range(config.signature_dimension))
        other = tuple(
            value if index < 24 else value + 10_000
            for index, value in enumerate(signature)
        )
        similarity = pair_similarity(signature, other, config)
        self.assertEqual(similarity["matches"], 24)
        self.assertEqual(similarity["similarity_units"], 7500)
        self.assertEqual(similarity["similarity_fixed"], "0.7500")
        self.assertTrue(similarity["boundary"])
        self.assertTrue(similarity["qualifies"])
        self.assertEqual(similarity_units(24, 32, 10_000, 1000), 7500)


if __name__ == "__main__":
    unittest.main()

