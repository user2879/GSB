from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal

from driftbench.attribution import compute_attribution_group
from driftbench.bucketing import bucket_hash, group_for_bucket
from driftbench.concurrency import ParallelRunner
from driftbench.drift import compute_drift_group
from driftbench.engine import DiagnosticEngine
from driftbench.loading import load_workspace
from driftbench.metrics import compute_metric
from driftbench.models import Config, Exposure, FilterCondition, Spec
from tests.common import SAMPLES


def config() -> Config:
    return Config(
        control_group="control",
        experiment_group="treatment",
        bucket_hash_seed="seed",
        bucket_count=8,
        active_spec_version="s1",
        fixed_decimal_places=6,
        quantization_step=1,
        concurrency=2,
        batch_size=8,
        snapshot_commit_interval_batches=1,
        bucket_property_key="bucket_key",
        fallback_policy="user_id_only",
        missing_attribute_group=None,
        drift_tolerance=Decimal("0.01"),
        min_affected_ratio=Decimal("0.05"),
        attribution_order=("truncation_rank", "normalization", "dedup_count", "sample_filter"),
        snapshot_version=1,
    )


def event(request_id: str, user_id: str, rank: int, clicked: int, event_ms: int = 100) -> Exposure:
    return Exposure(
        request_id=request_id,
        experiment_group="control",
        user_id=user_id,
        item_id="item-1",
        rank=rank,
        model_score=0.5,
        clicked=clicked,
        converted=0,
        event_ms=event_ms,
    )


def metric_stub(value_fixed: int, key: str) -> dict:
    return {
        "value_fixed": value_fixed,
        "value": f"{value_fixed / 1_000_000:.6f}",
        "cache_key": key,
        "numerator_weight": value_fixed,
        "denominator_weight": 1_000_000,
        "participant_count": 1,
    }


class MetricTests(unittest.TestCase):
    def test_truncation_dedup_filter_and_fixed_point(self) -> None:
        cfg = config()
        events = [
            event("r1", "u1", 1, 1),
            event("r2", "u1", 2, 0),
            event("r3", "u2", 6, 1),
            event("r4", "u3", 3, 1),
            event("r5", "u4", 1, 0, event_ms=200),
        ]
        spec = Spec("s1", "ctr", 5, "none", False)
        with ParallelRunner(2) as runner:
            base = compute_metric(events, "control", spec, cfg, runner, cache_key="base")
            dedup = compute_metric(
                events,
                "control",
                replace(spec, dedup_count=True),
                cfg,
                runner,
                cache_key="dedup",
            )
            filtered = compute_metric(
                events,
                "control",
                replace(spec, sample_filter=(FilterCondition("event_ms", "gte", 200),)),
                cfg,
                runner,
                cache_key="filtered",
            )
        self.assertEqual(base["participant_count"], 4)
        self.assertEqual(base["numerator_weight"], 2)
        self.assertEqual(base["value"], "0.500000")
        self.assertEqual(base["event_refs"], ["r1|item-1", "r2|item-1", "r4|item-1", "r5|item-1"])
        self.assertEqual(dedup["dedup_removed_count"], 1)
        self.assertEqual(dedup["value"], "0.666667")
        self.assertEqual(filtered["participant_count"], 1)
        self.assertEqual(filtered["event_refs"], ["r5|item-1"])
        self.assertEqual(filtered["value"], "0.000000")
        self.assertEqual(base["request_set_fingerprint"], base["request_set_fingerprint"])

    def test_bucket_assignment_is_deterministic_and_seed_sensitive(self) -> None:
        cfg = config()
        first = bucket_hash(cfg.bucket_hash_seed, "u1", "A", cfg.bucket_count)
        second = bucket_hash(cfg.bucket_hash_seed, "u1", "A", cfg.bucket_count)
        changed = bucket_hash("other", "u1", "A", cfg.bucket_count)
        self.assertEqual(first, second)
        self.assertEqual(group_for_bucket(first, cfg), group_for_bucket(second, cfg))
        self.assertNotEqual(first, changed)

    def test_exact_tolerance_is_boundary_and_not_significant(self) -> None:
        cfg = config()
        events = [event("r1", "u1", 1, 1), event("r2", "u2", 8, 0)]
        before = Spec("s1", "ctr", 10, "none", False)
        after = Spec("s2", "ctr", 5, "none", False)
        # 0.01 in six-decimal fixed units is exactly the configured tolerance.
        from_metric = metric_stub(100_000, "m1")
        to_metric = metric_stub(110_000, "m2")
        with ParallelRunner(2) as runner:
            drift = compute_drift_group(
                events,
                "control",
                before,
                after,
                from_metric,
                to_metric,
                cfg,
                runner,
                cache_key="drift",
            )
        self.assertTrue(drift["boundary"])
        self.assertFalse(drift["significant"])
        self.assertEqual(drift["delta_fixed"], 10_000)
        self.assertEqual(drift["tolerance_fixed"], 10_000)
        self.assertTrue(drift["boundary_samples"])
        self.assertEqual(drift["sources"]["truncation_rank"]["impacted_count"], 1)
        self.assertTrue(drift["sources"]["truncation_rank"]["changed"])

    def test_four_source_attribution_telescopes_to_total(self) -> None:
        state = load_workspace(SAMPLES)
        cfg = state.config
        from_spec = Spec("a", "ctr", 10, "none", False, ())
        to_spec = Spec(
            "b",
            "ctr",
            5,
            "rank_inverse",
            True,
            (FilterCondition("event_ms", "gte", 1_760_002_142_000),),
        )
        custom = state.clone()
        custom.specs = {"a": from_spec, "b": to_spec}
        custom.config = replace(cfg, active_spec_version="b")
        report = DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(custom)
        self.assertEqual(len(report["attribution"]["groups"]), 2)
        for group in report["attribution"]["groups"]:
            self.assertEqual(group["changed_sources"], [
                "truncation_rank",
                "normalization",
                "dedup_count",
                "sample_filter",
            ])
            self.assertEqual(group["residual_fixed"], 0)
            self.assertTrue(group["within_tolerance"])
            self.assertTrue(group["target_check_ok"])
            self.assertEqual(group["contribution_sum_fixed"], group["total_delta_fixed"])
            self.assertEqual(
                [row["source"] for row in group["contributions"]],
                ["truncation_rank", "normalization", "dedup_count", "sample_filter"],
            )


if __name__ == "__main__":
    unittest.main()
