from __future__ import annotations

import copy
import unittest

from meshwork.stats import build_stats, floor_fixed, nearest_rank_p95


def outcome(
    request_id: str,
    service: str,
    seq: int,
    final: str,
    duration: int,
    completed: int,
    error: str | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "request_seq": seq,
        "service": service,
        "final_outcome": final,
        "error_code": error,
        "attempt_count": 1,
        "total_duration_ms": duration,
        "completed_at_wall_ms": completed,
    }


def attempt(request_id: str, service: str, seq: int, attempt_id: int, outcome_name: str) -> dict:
    return {
        "attempt_id": attempt_id,
        "request_id": request_id,
        "request_seq": seq,
        "attempt_no": 1,
        "service": service,
        "outcome": outcome_name,
        "started_wall_ms": 1000 + seq,
    }


class StatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scale = 1_000_000
        self.outcomes = [
            outcome("a1", "alpha", 1, "success", 10, 1100),
            outcome("a2", "alpha", 2, "success", 20, 1110),
            outcome("a3", "alpha", 3, "exhausted", 30, 1120, "E_EXHAUST"),
            outcome("b1", "beta", 4, "success", 15, 1115),
            outcome("b2", "beta", 5, "success", 25, 1125),
            outcome("b3", "beta", 6, "exhausted", 35, 1135, "E_EXHAUST"),
        ]
        self.attempts = [
            attempt("a1", "alpha", 1, 1, "success"),
            attempt("a2", "alpha", 2, 2, "error"),
            attempt("a2", "alpha", 2, 3, "success"),
            attempt("a3", "alpha", 3, 4, "error"),
            attempt("b1", "beta", 4, 5, "success"),
            attempt("b2", "beta", 5, 6, "success"),
            attempt("b3", "beta", 6, 7, "error"),
        ]

    def test_fixed_point_formulas_and_p95(self) -> None:
        stats = build_stats(self.outcomes, self.attempts, self.scale)
        alpha = next(row for row in stats["services"] if row["service"] == "alpha")
        self.assertEqual(alpha["success_ratio_fixed"], floor_fixed(2, 3, self.scale))
        self.assertEqual(alpha["success_ratio_fixed"], 666666)
        self.assertEqual(alpha["p95_latency_ms"], 30)
        self.assertEqual(alpha["p95_latency_ms_fixed"], 30 * self.scale)
        self.assertEqual(stats["method"]["rounding"], "non_negative_floor_truncation")
        self.assertEqual(stats["method"]["comparison_error_bound"]["ratio_scaled_units"], 2)
        self.assertEqual(stats["method"]["comparison_error_bound"]["p95_scaled_units"], 2 * self.scale)
        self.assertEqual(nearest_rank_p95([3, 1, 2]), 3)

    def test_equal_ratios_are_tied_and_marked_with_stable_service_order(self) -> None:
        stats = build_stats(self.outcomes, self.attempts, self.scale)
        ranking = stats["rankings"]["success_ratio"]
        self.assertEqual([row["service"] for row in ranking], ["alpha", "beta"])
        self.assertTrue(all(row["within_error_bound"] for row in ranking))
        self.assertTrue(all(row["tie_break"] == "service_id_asc" for row in ranking))
        comparisons = stats["pairwise_comparisons"]["success_ratio"]
        self.assertTrue(comparisons[0]["within_error_bound"])
        self.assertEqual(comparisons[0]["winner_by_stable_rule"], "alpha")

    def test_attribution_is_independent_of_input_order(self) -> None:
        first = build_stats(self.outcomes, self.attempts, self.scale)
        shuffled_outcomes = copy.deepcopy(list(reversed(self.outcomes)))
        shuffled_attempts = copy.deepcopy(list(reversed(self.attempts)))
        second = build_stats(shuffled_outcomes, shuffled_attempts, self.scale)
        self.assertEqual(first["services"], second["services"])
        self.assertEqual(first["rankings"], second["rankings"])
        self.assertEqual(first["overall"], second["overall"])


if __name__ == "__main__":
    unittest.main()
