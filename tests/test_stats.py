"""Fixed-point statistics tests."""

import unittest

from workbench.stats import aggregate


class StatsTest(unittest.TestCase):
    def test_mean_truncates_toward_zero(self):
        agg = aggregate([1, 2])
        self.assertEqual(agg["mean_x1000"], 1500)
        agg = aggregate([1, 1, 2])
        self.assertEqual(agg["mean_x1000"], 1333)

    def test_p95_nearest_rank(self):
        agg = aggregate(list(range(1, 101)))
        self.assertEqual(agg["p95_x1000"], 95000)
        self.assertEqual(agg["max_x1000"], 100000)

    def test_small_sample(self):
        agg = aggregate([7])
        self.assertEqual(agg["p95_x1000"], 7000)
        self.assertEqual(agg["max_x1000"], 7000)

    def test_empty(self):
        agg = aggregate([])
        self.assertEqual(agg["count"], 0)
        self.assertEqual(agg["mean_x1000"], 0)

    def test_metadata(self):
        agg = aggregate([1, 2, 3])
        self.assertEqual(agg["scale"], 1000)
        self.assertEqual(agg["error_bound_x1000"], 1)
        self.assertEqual(agg["truncation"], "toward_zero")


if __name__ == "__main__":
    unittest.main()
