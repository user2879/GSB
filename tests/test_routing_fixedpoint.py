from __future__ import annotations

import random
import unittest

from gray_merge_lab.errors import LabError
from gray_merge_lab.fixedpoint import (
    BatchFixedStats,
    aggregate_batch_stats,
    compare_candidates,
    q_to_original_interval,
    raw_duration_to_q,
    trunc_div,
    validate_scale_exp,
)
from gray_merge_lab.routing import batch_partition_digest, plan_batches, route_sample
from gray_merge_lab.specs import load_spec
from tests.helpers import test_spec


class RoutingTests(unittest.TestCase):
    def test_bucket_domain_and_stability(self):
        spec = test_spec(weight_bp=5000)
        sample = spec.samples[0]
        rule = spec.rules[0]
        routed = [route_sample(sample, rule) for _ in range(100)]
        self.assertTrue(1 <= routed[0].bucket <= 10000)
        self.assertEqual(len({item.bucket for item in routed}), 1)
        self.assertEqual(len({item.destination for item in routed}), 1)
        bucket = routed[0].bucket
        stable_rule = rule.__class__(**{**rule.__dict__, "canary_weight_bp": bucket - 1})
        canary_rule = rule.__class__(**{**rule.__dict__, "canary_weight_bp": bucket})
        self.assertEqual(route_sample(sample, stable_rule).destination, "stable")
        self.assertEqual(route_sample(sample, canary_rule).destination, "canary")

    def test_weight_boundaries(self):
        sample = test_spec(weight_bp=0).samples[0]
        rule = test_spec(weight_bp=0).rules[0]
        zero = rule.__class__(**{**rule.__dict__, "canary_weight_bp": 0})
        full = rule.__class__(**{**rule.__dict__, "canary_weight_bp": 10000})
        self.assertEqual(route_sample(sample, zero).destination, "stable")
        self.assertEqual(route_sample(sample, full).destination, "canary")

    def test_batch_partition_is_order_independent(self):
        spec = test_spec(sample_count=9, batch_size=3)
        canonical = plan_batches(spec.samples, spec.rules)
        digest = batch_partition_digest(canonical)
        for seed in range(10):
            shuffled = list(spec.samples)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(batch_partition_digest(plan_batches(shuffled, spec.rules)), digest)
        self.assertEqual([len(item.samples) for item in canonical], [3, 3, 3])
        self.assertTrue(all(len(item.samples) <= 3 for item in canonical))

    def test_batches_are_client_homogeneous(self):
        spec = test_spec(sample_count=4, batch_size=2)
        plans = plan_batches(spec.samples, spec.rules)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(len({item.client_key for item in plan.samples}), 1)
            self.assertEqual(len(set(plan.buckets)), 1)

    def test_local_data_covers_both_destinations_and_rules(self):
        root = __import__("pathlib").Path(__file__).resolve().parent.parent / "gray_merge_lab" / "data"
        spec = load_spec(root / "backends.json", root / "rules.json", root / "samples.ndjson")
        plans = plan_batches(spec.samples, spec.rules)
        self.assertEqual(len(plans), 8)
        self.assertEqual({item.rule_id for item in plans}, {"search", "recommend"})
        self.assertEqual({item.destination for item in plans}, {"stable", "canary"})
        self.assertEqual(sum(len(item.samples) for item in plans), 28)


class FixedPointTests(unittest.TestCase):
    def test_scale_and_raw_values(self):
        stats = BatchFixedStats("b1", 7, 4, 3)
        self.assertEqual(stats.duration_q, 7000)
        self.assertEqual(stats.samples_q, 4000)
        self.assertEqual(stats.single_value_error_original, 0.001)
        self.assertEqual(raw_duration_to_q(7, 3), 7000)
        self.assertEqual(q_to_original_interval(7000, 3), (7.0, 7.001))

    def test_aggregate_stats_and_error_bounds(self):
        aggregate = aggregate_batch_stats(
            [BatchFixedStats("b2", 6, 4, 3), BatchFixedStats("b1", 5, 2, 3)]
        )
        self.assertEqual(aggregate.batch_ids, ("b1", "b2"))
        self.assertEqual(aggregate.raw_duration_sum_ms, 11)
        self.assertEqual(aggregate.raw_sample_sum, 6)
        self.assertEqual(aggregate.avg_duration_q, 5500)
        self.assertEqual(aggregate.avg_samples_q, 3000)
        self.assertEqual(aggregate.single_value_error_original, 0.001)
        self.assertEqual(aggregate.batch_aggregate_error_original, 0.002)
        self.assertEqual(aggregate.canonical_aggregate_error_original, 0.001)
        interval = q_to_original_interval(aggregate.avg_duration_q, 3)
        self.assertLessEqual(interval[0], aggregate.raw_duration_sum_ms / aggregate.batch_count)
        self.assertLessEqual(aggregate.raw_duration_sum_ms / aggregate.batch_count, interval[1])

    def test_aggregate_rejects_mixed_scales(self):
        with self.assertRaises(LabError) as caught:
            aggregate_batch_stats([BatchFixedStats("b1", 1, 1, 2), BatchFixedStats("b2", 1, 1, 3)])
        self.assertEqual(caught.exception.code, "E_FORMAT")

    def test_candidate_within_bound_uses_batch_id_tiebreak(self):
        decision = compare_candidates("batch-z", 12346, "batch-a", 12345)
        self.assertTrue(decision.indistinguishable)
        self.assertTrue(decision.error_affected)
        self.assertEqual(decision.winner, "batch-a")
        self.assertEqual(decision.combined_error_bound_q, 2)

    def test_candidate_outside_bound_uses_lower_duration(self):
        decision = compare_candidates("batch-z", 12000, "batch-a", 12003)
        self.assertFalse(decision.indistinguishable)
        self.assertFalse(decision.error_affected)
        self.assertEqual(decision.winner, "batch-z")

    def test_truncation_and_validation(self):
        self.assertEqual(trunc_div(7, 3), 2)
        self.assertEqual(trunc_div(-7, 3), -2)
        with self.assertRaises(LabError) as caught:
            validate_scale_exp(10)
        self.assertEqual(caught.exception.code, "E_SPEC")


if __name__ == "__main__":
    unittest.main()

