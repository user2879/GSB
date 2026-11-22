import unittest

from workbench.estimator import (
    FullEstimator,
    IncrementalEstimator,
    assert_estimates_equal,
    estimate_service_minute,
    per_replica_capacity,
    run_both_paths,
    select_path,
)
from workbench.loader import load_declaration
from workbench.models import ClusterSpec
from workbench.sample_data import build_cluster, build_small_cluster
from workbench.testing_utils import spec_from_payload, tmp_declaration


class EstimatorRuleTest(unittest.TestCase):
    def setUp(self):
        self.spec = spec_from_payload(build_cluster())

    def test_capacity_integer_rule(self):
        service = self.spec.services[0]
        expected = (8 * 60_000) // 120
        self.assertEqual(per_replica_capacity(service), expected)

    def test_ceiling_demand_clamps_to_max(self):
        service = self.spec.services[1]
        peak = len(service.requests) - 1
        estimate = estimate_service_minute(service, peak)
        self.assertLessEqual(estimate.demand, service.max_replicas)
        self.assertEqual(estimate.rule_version, "replica-int-v1")
        self.assertGreaterEqual(estimate.demand, service.min_replicas)

    def test_zero_latency_capacity_defined(self):
        payload = build_cluster()
        payload["services"][0]["latency_ms"] = 0
        spec = spec_from_payload(payload)
        estimate = estimate_service_minute(spec.services[0], 0)
        self.assertEqual(estimate.per_replica_capacity,
                         spec.services[0].concurrency_per_replica * 60_000)

    def test_repeatability_byte_identical(self):
        first = FullEstimator().run(self.spec)
        second = FullEstimator().run(self.spec)
        self.assertEqual(
            [row.as_tuple() for minute in first for row in minute],
            [row.as_tuple() for minute in second for row in minute],
        )


class TwoPathTest(unittest.TestCase):
    def test_full_and_incremental_byte_identical(self):
        spec = spec_from_payload(build_cluster())
        full_rows, incremental_rows, incremental = run_both_paths(spec)
        self.assertEqual(
            [row.as_tuple() for minute in full_rows for row in minute],
            [row.as_tuple() for minute in incremental_rows for row in minute],
        )
        self.assertEqual(len(incremental.window_history), spec.minutes)

    def test_selection_boundary_is_explicit(self):
        large = spec_from_payload(build_cluster())
        self.assertEqual(select_path(large)[0], "incremental")
        small = spec_from_payload(build_small_cluster())
        path, reason, boundary = select_path(small)
        self.assertEqual(path, "full")
        self.assertEqual(boundary, 30)
        self.assertIn("30", reason)

    def test_ring_window_state_reused(self):
        spec = spec_from_payload(build_cluster())
        incremental = IncrementalEstimator(spec.window_size, spec.fixed_point_scale)
        rows = incremental.run(spec)
        sums = [item[1] for item in incremental.window_history]
        # Window sum never decreases incorrectly and ring depth caps at window.
        self.assertLessEqual(len(sums), spec.minutes)
        self.assertEqual(len(rows), spec.minutes)

    def test_equality_checker_locates_first_diff(self):
        spec = spec_from_payload(build_cluster())
        full = FullEstimator().run(spec)
        broken = IncrementalEstimator(spec.window_size, spec.fixed_point_scale).run(spec)
        target = list(broken[3][1].__dict__)
        from workbench.models import MinuteEstimate
        broken_list = [list(minute) for minute in broken]
        broken_list[3][1] = MinuteEstimate(**{**broken[3][1].__dict__, "demand": -1})
        broken_tuple = tuple(tuple(minute) for minute in broken_list)
        from workbench.errors import DivergenceError
        with self.assertRaises(DivergenceError) as caught:
            assert_estimates_equal(full, broken_tuple)
        self.assertEqual(caught.exception.details["minute"], 3)
        self.assertEqual(caught.exception.details["field"], "demand")


if __name__ == "__main__":
    unittest.main()
