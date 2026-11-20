from __future__ import annotations

import math
import unittest

from scorecard_app.monotonic import solve_monotonic
from scorecard_app.woe import compute_woe


def _bins() -> list[dict[str, object]]:
    return [
        {"index": 0, "lower": 0, "upper": 9, "count": 100, "good": 20, "bad": 80, "is_missing": False},
        {"index": 1, "lower": 10, "upper": 19, "count": 100, "good": 50, "bad": 50, "is_missing": False},
        {"index": 2, "lower": 20, "upper": 29, "count": 100, "good": 80, "bad": 20, "is_missing": False},
    ]


class WoeTests(unittest.TestCase):
    def test_zero_frequency_is_smoothed_and_iv_contributions_sum(self) -> None:
        bins = [
            {"index": 0, "lower": 0, "upper": 9, "count": 10, "good": 10, "bad": 0, "is_missing": False},
            {"index": 1, "lower": 10, "upper": 19, "count": 10, "good": 0, "bad": 10, "is_missing": False},
        ]
        result = compute_woe("x", "numeric", bins, 0.5)
        self.assertEqual(result["smoothed_bin_count"], 2)
        self.assertTrue(all(item["smoothed"] for item in result["bins"]))
        self.assertTrue(all(math.isfinite(item["woe"]) for item in result["bins"]))
        self.assertAlmostEqual(
            result["iv"],
            round(sum(item["iv_contribution"] for item in result["bins"]), 12),
            places=12,
        )

    def test_repeated_woe_is_identical(self) -> None:
        first = compute_woe("x", "numeric", _bins(), 0.5)
        second = compute_woe("x", "numeric", _bins(), 0.5)
        self.assertEqual(first, second)


class MonotonicTests(unittest.TestCase):
    def test_ascending_conflicts_merge_until_satisfied(self) -> None:
        result = solve_monotonic(
            "x",
            "numeric",
            _bins(),
            "ascending",
            ["max_gap", "left", "right"],
            20,
            0.5,
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(len(result["corrections"]), 1)
        nonmissing = [item for item in result["bins"] if not item["is_missing"]]
        for left, right in zip(nonmissing, nonmissing[1:]):
            self.assertLessEqual(left["woe"], right["woe"])
        correction = result["corrections"][0]
        self.assertIn("merged_bounds", correction)
        self.assertIn("before_woe", correction)
        self.assertIn("after_woe", correction)
        self.assertIn("hit_rule", correction)

    def test_descending_is_already_monotonic(self) -> None:
        result = solve_monotonic(
            "x", "numeric", _bins(), "descending", ["max_gap", "left"], 20, 0.5
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["corrections"], [])

    def test_unresolved_preserves_current_state(self) -> None:
        result = solve_monotonic(
            "x", "numeric", _bins(), "ascending", ["max_gap", "left"], 1, 0.5
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(len(result["corrections"]), 1)
        self.assertTrue(result["unresolved_violations"])
        self.assertEqual(len(result["bins"]), 2)
        self.assertTrue(all("woe" in item for item in result["bins"]))


if __name__ == "__main__":
    unittest.main()
