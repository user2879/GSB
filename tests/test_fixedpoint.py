import unittest

from gpu_scheduler import fixedpoint as fp


class FixedPointTests(unittest.TestCase):
    def test_ratio_truncates_toward_zero_with_bound_one(self):
        stat = fp.ratio_stat(1, 3, 10000)
        self.assertEqual(stat.value, 3333)
        self.assertEqual(stat.abs_error_bound, 1)
        self.assertEqual(stat.truncation, fp.TRUNC_TOWARD_ZERO)

    def test_average_uses_floor_truncation(self):
        stat = fp.average_stat(10, 3, 100)
        self.assertEqual(stat.value, 333)
        self.assertEqual(stat.truncation, fp.TRUNC_FLOOR)
        self.assertEqual(stat.abs_error_bound, 1)

    def test_zero_denominator_is_defined_as_zero(self):
        self.assertEqual(fp.quantize_ratio(1, 0, 10000), 0)
        self.assertEqual(fp.average_stat(5, 0, 10000).value, 0)

    def test_comparison_flags_error_band_and_breaks_ties_stably(self):
        left = fp.ratio_stat(5000, 10000, 10000)
        right = fp.ratio_stat(5001, 10000, 10000)
        comparison = fp.compare_ratios(left, right, "pool-b", "pool-a")
        self.assertTrue(comparison.within_error)
        self.assertEqual(comparison.combined_error_bound, 2)
        self.assertTrue(comparison.tie_break_applied)

    def test_comparison_outside_band_is_deterministic_order(self):
        left = fp.ratio_stat(9000, 10000, 10000)
        right = fp.ratio_stat(1000, 10000, 10000)
        comparison = fp.compare_ratios(left, right, "pool-a", "pool-b")
        self.assertFalse(comparison.within_error)
        self.assertEqual(comparison.order, 1)

    def test_aggregate_keeps_single_unit_error_bound(self):
        aggregate = fp.aggregate_ratio([(1, 3), (2, 3)], 10000)
        self.assertEqual(aggregate.value, 5000)
        self.assertEqual(aggregate.abs_error_bound, 1)

    def test_rank_is_stable_across_calls(self):
        items = [
            ("pool-c", fp.ratio_stat(30, 100, 10000)),
            ("pool-a", fp.ratio_stat(30, 100, 10000)),
            ("pool-b", fp.ratio_stat(70, 100, 10000)),
        ]
        first = fp.rank(items)
        second = fp.rank(items)
        self.assertEqual([row["key"] for row in first],
                         [row["key"] for row in second])
        self.assertEqual(first[0]["key"], "pool-b")
        # Exact-equal pools break on the identifier, not hash order.
        equal = [row for row in first if row["stat"]["value"] == 3000]
        self.assertEqual([row["key"] for row in equal],
                         ["pool-a", "pool-c"])

    def test_repeated_results_are_identical_integers(self):
        values = {
            fp.quantize_ratio(7, 11, 100000) for _ in range(20)
        }
        self.assertEqual(len(values), 1)


if __name__ == "__main__":
    unittest.main()
