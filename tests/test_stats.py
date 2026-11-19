from __future__ import annotations

import unittest
from dataclasses import replace

from bucketpack.engine import (
    assert_merge_consistency,
    direct_statistics,
    merge_pack_summaries,
    run_full,
)
from bucketpack.errors import E_DIVERGE, BusinessError

from .common import sample_spec


class StatisticsTests(unittest.TestCase):
    def test_merged_pack_stats_equal_direct_stats(self) -> None:
        spec = sample_spec()
        state = run_full(spec)
        merged = merge_pack_summaries(spec, state.packs)
        direct = direct_statistics(spec, state.expansions)
        self.assertEqual(merged, direct)
        self.assertEqual(state.result["merged"], state.result["direct"])
        for source_id, source_stat in merged["sources"].items():
            self.assertEqual(
                source_stat["row_count"],
                state.expansions[source_id].rows,
            )
            self.assertEqual(
                source_stat["width_used"],
                source_stat["row_count"] * state.expansions[source_id].row_width,
            )

    def test_divergence_reports_first_bucket_and_pack(self) -> None:
        spec = sample_spec()
        state = run_full(spec)
        direct = direct_statistics(spec, state.expansions)
        pack = state.packs[0]
        bucket_id = min(pack.bucket_counts)
        counts = dict(pack.bucket_counts)
        counts[bucket_id] = list(counts[bucket_id])
        counts[bucket_id][0] += 1
        tampered = replace(pack, bucket_counts=counts)
        packs = [tampered if item.pack_id == pack.pack_id else item for item in state.packs]
        merged = merge_pack_summaries(spec, packs)
        with self.assertRaises(BusinessError) as caught:
            assert_merge_consistency(merged, direct, packs)
        error = caught.exception
        self.assertEqual(error.code, E_DIVERGE)
        self.assertEqual(error.details["first_differing_pack_id"], pack.pack_id)
        self.assertEqual(error.details["first_differing_bucket_id"], bucket_id)

    def test_derived_statistics_are_sum_of_bucket_counts(self) -> None:
        spec = sample_spec()
        state = run_full(spec)
        for stat in state.result["derived_statistics"].values():
            expected = sum(
                sum(counts) for counts in stat["per_bucket"].values()
            )
            self.assertEqual(stat["value"], expected)


if __name__ == "__main__":
    unittest.main()
