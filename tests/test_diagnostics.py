import unittest

from lease_diag.diagnostics import (
    DEFAULT_SCALE, comparison_error_bound, fixed_ratio, global_watermark,
    pool_watermark, rank_pools,
)
from lease_diag.model import Pool


def make_pool(pid, capacity, priority=0):
    return Pool(pid, capacity, 100, 20, priority, capacity)


class FixedPointTests(unittest.TestCase):
    def test_truncation_toward_zero(self):
        # 1/3 * 10000 = 3333.33 -> truncated to 3333.
        self.assertEqual(fixed_ratio(1, 3), 3333)
        self.assertEqual(fixed_ratio(2, 3), 6666)

    def test_single_value_quantization_error_one_ulp(self):
        mark = pool_watermark(make_pool("P1", 3), 1, 0)
        self.assertEqual(mark.ratio_fp, 3333)
        self.assertEqual(mark.value_error_ulp, 1)
        self.assertAlmostEqual(mark.value_error_ratio, 1.0 / DEFAULT_SCALE)
        self.assertEqual(mark.truncation, "toward_zero")

    def test_global_watermark_aggregates_pools(self):
        pools = {"P1": make_pool("P1", 2), "P2": make_pool("P2", 4)}
        overall = global_watermark(pools, {"P1": 2, "P2": 1},
                                   {"P1": 0, "P2": 2})
        self.assertEqual(overall.capacity, 6)
        self.assertEqual(overall.effective_leases, 3)
        self.assertEqual(overall.queue_length, 2)
        self.assertEqual(overall.ratio_fp, 5000)

    def test_comparison_error_bound_is_two_ulp(self):
        bound = comparison_error_bound()
        self.assertEqual(bound["comparison_error_ulp"], 2)
        self.assertAlmostEqual(bound["comparison_error_ratio"],
                               2.0 / DEFAULT_SCALE)

    def test_ranking_flags_error_influenced_close_pools(self):
        # Capacities chosen so ratios differ by <= 2 ulp at scale 10000.
        marks = [
            pool_watermark(make_pool("P1", 10000), 1, 0),  # 1
            pool_watermark(make_pool("P2", 5000), 1, 0),   # 2
        ]
        ranking = rank_pools(marks)
        self.assertEqual([row["pool_id"] for row in ranking], ["P2", "P1"])
        self.assertFalse(ranking[0]["error_influenced"])
        self.assertTrue(ranking[1]["error_influenced"])
        self.assertEqual(ranking[1]["difference_from_previous_ulp"], 1)

    def test_ranking_stable_when_identical_ratio(self):
        marks = [
            pool_watermark(make_pool("P_Z", 2), 1, 0),
            pool_watermark(make_pool("P_A", 2), 1, 0),
        ]
        ranking = rank_pools(marks)
        # Same ratio: stable tie-break by pool id, flagged as error influenced.
        self.assertEqual([row["pool_id"] for row in ranking],
                         ["P_A", "P_Z"])
        self.assertTrue(all(row["error_influenced"] for row in ranking[1:]))


if __name__ == "__main__":
    unittest.main()
