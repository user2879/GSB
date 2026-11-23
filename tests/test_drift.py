from __future__ import annotations

import unittest
from copy import deepcopy

from drift_diag.canonical import canonical_json
from drift_diag.config import EvaluationConfig
from drift_diag.drift import (
    INT64_MAX,
    INT64_MIN,
    aggregate_all,
    compare_field,
    quantize_difference,
    saturate_int64,
    truncate_toward_zero,
)
from drift_diag.scheduler import schedule_decisions

from .helpers import complete_actual, config, decisions_for, target_v3


class DriftComparisonTests(unittest.TestCase):
    def setUp(self):
        self.target = target_v3()
        self.actual = complete_actual(self.target)

    def test_missing_observation_is_a_drift(self):
        actual = deepcopy(self.actual)
        del actual["worker"]["probe_ms"]
        decision = compare_field(self.target.workloads["worker"], "probe_ms", actual, scale_exp=3)
        self.assertTrue(decision["drift"])
        self.assertEqual(decision["reason"], "missing_observation")
        self.assertFalse(decision["observed_present"])
        self.assertIsNone(decision["fixed_metric"])

    def test_integer_fixed_point_delta_and_basis(self):
        actual = deepcopy(self.actual)
        actual["api-gateway"]["resource_limit"] = 1600
        decision = compare_field(
            self.target.workloads["api-gateway"], "resource_limit", actual, scale_exp=3
        )
        self.assertTrue(decision["drift"])
        self.assertEqual(decision["basis"], "integer_fixed_point")
        self.assertEqual(decision["fixed_metric"]["fixed_value"], -64000)
        self.assertEqual(decision["fixed_metric"]["absolute_fixed"], 64000)
        self.assertEqual(decision["fixed_metric"]["quantization_error_upper_units"], 0)

    def test_results_are_byte_stable_and_layout_independent(self):
        first, _ = schedule_decisions(self.target, self.actual, config(global_limit=2))
        second, _ = schedule_decisions(self.target, self.actual, config(global_limit=5, per_workload_quota=2))
        self.assertEqual(canonical_json(first), canonical_json(second))
        third = decisions_for(self.target, self.actual)
        self.assertEqual(canonical_json(first), canonical_json(third))

    def test_boundary_ratio_uses_inclusive_stable_tie_break(self):
        actual = deepcopy(self.actual)
        actual["cache"]["probe_ms"] = 1100
        decisions = decisions_for(self.target, actual)
        metrics = aggregate_all(
            self.target,
            decisions,
            EvaluationConfig(tolerance_num=1, tolerance_den=6).validate(),
        )
        cache = metrics["workloads"]["cache"]
        self.assertEqual(cache["drift_field_count"], 1)
        self.assertTrue(cache["converged"])
        self.assertTrue(cache["boundary_sensitive"])
        self.assertTrue(cache["boundary_rule_applied"])
        self.assertTrue(cache["error_affected"])
        self.assertEqual(cache["workload_error_upper_units"], 0)
        self.assertTrue(cache["exact_integer_quantization"])

    def test_metric_overflow_saturates_and_truncates_toward_zero(self):
        self.assertEqual(quantize_difference(10**20, 0, 18), INT64_MAX)
        self.assertEqual(quantize_difference(0, 10**20, 18), INT64_MIN)
        self.assertEqual(truncate_toward_zero(19, 10), 1)
        self.assertEqual(truncate_toward_zero(-19, 10), -1)
        self.assertEqual(saturate_int64(INT64_MAX + 1), INT64_MAX)

    def test_non_numeric_drift_is_counted_but_not_in_numeric_metric(self):
        actual = deepcopy(self.actual)
        actual["worker"]["labels"] = {"app": "other"}
        decisions = decisions_for(self.target, actual)
        metrics = aggregate_all(self.target, decisions, EvaluationConfig())
        worker = metrics["workloads"]["worker"]
        self.assertEqual(worker["drift_fields"], ["labels"])
        self.assertEqual(worker["numeric_drift_metric_units"], 0)
        self.assertFalse(worker["converged"])


if __name__ == "__main__":
    unittest.main()
