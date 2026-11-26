import unittest

from intent_router.statistics import (
    fixed_point_distribution,
    permille_share,
)


class StatisticsTest(unittest.TestCase):
    def test_distribution_counts(self):
        dist = fixed_point_distribution(
            ["A", "B", "C"], ["A", "A", "B", "C", "A"]
        )
        self.assertEqual(dist, {"A": 3, "B": 1, "C": 1})

    def test_permille_share_sums_to_one_thousand_or_less(self):
        shares = permille_share({"A": 1, "B": 1, "C": 2})
        self.assertEqual(shares["C"], 500)
        self.assertLessEqual(sum(shares.values()), 1000)

    def test_empty_distribution(self):
        self.assertEqual(permille_share({"A": 0}), {"A": 0})


if __name__ == "__main__":
    unittest.main()
