import unittest

import numpy as np

from shardcast.sample import (
    aggregate_rows,
    canonical_json,
    generate_sample,
    merge_aggregate,
    sample_bytes,
)


class SampleDeterminismTests(unittest.TestCase):
    def test_sample_bytes_bit_identical_across_calls(self) -> None:
        first = sample_bytes(42, "shard-x", 257)
        second = sample_bytes(42, "shard-x", 257)
        self.assertEqual(first, second)

    def test_distinct_seed_or_shard_changes_bytes(self) -> None:
        base = sample_bytes(1, "s", 64)
        other_seed = sample_bytes(2, "s", 64)
        other_shard = sample_bytes(1, "t", 64)
        self.assertNotEqual(base, other_seed)
        self.assertNotEqual(base, other_shard)

    def test_value_range_and_group_range(self) -> None:
        data = generate_sample(5, "range", 500)
        self.assertTrue(((data[:, 0] >= 0) & (data[:, 0] < 16)).all())
        self.assertTrue(((data[:, 1] >= -1000) & (data[:, 1] <= 1000)).all())

    def test_participation_excludes_negative_values(self) -> None:
        data = generate_sample(5, "range", 500)
        summary = aggregate_rows(data)
        expected = int((data[:, 1] >= 0).sum())
        self.assertEqual(summary["participating_rows"], expected)
        group_count_sum = sum(group["count"] for group in summary["groups"].values())
        self.assertEqual(group_count_sum, expected)

    def test_batching_does_not_change_aggregate(self) -> None:
        data = generate_sample(11, "batch", 333)
        full = aggregate_rows(data)
        merged = None
        for size in (1, 7, 31, 100):
            partial_merged = None
            for start in range(0, data.shape[0], size):
                partial = aggregate_rows(data[start:start + size])
                partial_merged = (
                    partial
                    if partial_merged is None
                    else merge_aggregate(partial_merged, partial)
                )
            self.assertEqual(canonical_json(partial_merged), canonical_json(full))
            merged = partial_merged
        self.assertEqual(merged, full)

    def test_merge_is_order_independent(self) -> None:
        data = generate_sample(13, "order", 200)
        left = aggregate_rows(data[:120])
        right = aggregate_rows(data[120:])
        self.assertEqual(
            canonical_json(merge_aggregate(left, right)),
            canonical_json(merge_aggregate(right, left)),
        )

    def test_group_stats_match_brute_force(self) -> None:
        data = generate_sample(17, "brute", 180)
        summary = aggregate_rows(data)
        participating = data[data[:, 1] >= 0]
        for group_id, stats in summary["groups"].items():
            values = participating[participating[:, 0] == int(group_id), 1]
            self.assertEqual(stats["count"], int(values.shape[0]))
            self.assertEqual(stats["sum"], int(values.sum(dtype=np.int64)))
            self.assertEqual(stats["min"], int(values.min()))
            self.assertEqual(stats["max"], int(values.max()))


if __name__ == "__main__":
    unittest.main()
