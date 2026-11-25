from __future__ import annotations

import unittest

from speaker_diag.metrics import (
    FixedPointPolicy,
    ScoredCandidate,
    attribution_diagnostics,
    stable_choose,
    worker_watermark_decisions,
)
from speaker_diag.pipeline import normalize_local_results
from speaker_diag.samples import sample_paths
from speaker_diag.spec import load_workload
from speaker_diag.synthetic import synthesize


class MetricTests(unittest.TestCase):
    def test_fixed_point_bounds_are_explicit(self) -> None:
        policy = FixedPointPolicy(scale=10000)
        data = policy.to_dict()
        self.assertEqual(data["single_value_error_bound"], "<1/10000")
        self.assertEqual(data["comparison_error_bound"], "<2/10000")
        self.assertEqual(data["minimum_distinguishable_difference_units"], 3)
        self.assertEqual(data["truncation"], "floor_non_negative")

    def test_error_bound_causes_stable_lexicographic_tie_break(self) -> None:
        decision = stable_choose(
            [ScoredCandidate("B002", 100), ScoredCandidate("B001", 99)],
            error_bound_q=2,
        )
        self.assertTrue(decision.affected_by_error_bound)
        self.assertEqual(decision.selected_key, "B001")
        strict = stable_choose(
            [ScoredCandidate("B002", 103), ScoredCandidate("B001", 100)],
            error_bound_q=2,
        )
        self.assertFalse(strict.affected_by_error_bound)
        self.assertEqual(strict.selected_key, "B002")

    def test_worker_watermark_tie_is_marked_and_stable(self) -> None:
        snapshot = {
            "workers": [
                {"worker_id": "W02", "lease_watermark_q": 5000, "concurrency_watermark_q": 5000, "active_leases": 1, "active_concurrency": 1},
                {"worker_id": "W01", "lease_watermark_q": 5000, "concurrency_watermark_q": 5000, "active_leases": 1, "active_concurrency": 1},
            ]
        }
        report = worker_watermark_decisions(snapshot, FixedPointPolicy(scale=10000))
        self.assertTrue(report["group_decision"]["affected_by_error_bound"])
        self.assertEqual(report["group_decision"]["selected_key"], "W01")

    def test_error_cause_proportions_sum_to_10000(self) -> None:
        spec = load_workload(sample_paths("samples")["v1"])
        bundle = synthesize(spec)
        _, normalized = normalize_local_results(spec, bundle)
        report = attribution_diagnostics(normalized, bundle, spec, FixedPointPolicy(scale=10000))
        proportions = report["error_causes"]["proportion_bps"]
        self.assertEqual(sum(proportions.values()), 10000)
        self.assertEqual(
            set(proportions),
            {"quantization_truncation", "overlap_attribution", "batch_reorder"},
        )


if __name__ == "__main__":
    unittest.main()
