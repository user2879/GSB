from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autoscale_diag.errors import BusinessError, E_SPEC
from autoscale_diag.spec import load_spec, parse_spec

from tests.helpers import SPEC_V1, mutated_sample, sample_payload


class SpecValidationTests(unittest.TestCase):
    def assert_spec_error(self, payload, field_path: str) -> None:
        with self.assertRaises(BusinessError) as context:
            parse_spec(payload)
        self.assertEqual(context.exception.code, E_SPEC)
        self.assertEqual(context.exception.field_path, field_path)

    def test_valid_sample_and_versions_are_stable(self) -> None:
        spec = load_spec(SPEC_V1)
        self.assertEqual(spec.declaration_version, "demo-v1")
        self.assertEqual(spec.minute_count, 12)
        self.assertEqual(len(spec.sha256()), 64)
        self.assertEqual(spec.sha256(), load_spec(SPEC_V1).sha256())
        self.assertEqual([item.id for item in spec.workloads], ["checkout", "payments", "search"])

    def test_duplicate_workload_id_rejected(self) -> None:
        payload = mutated_sample(lambda data: data["workloads"][1].__setitem__("id", "checkout"))
        self.assert_spec_error(payload, "workloads[1].id")

    def test_non_continuous_minute_rejected(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["observations"][3].__setitem__("minute", 5)
        )
        self.assert_spec_error(payload, "workloads[0].observations[3].minute")

    def test_duplicate_minute_rejected_with_field_path(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["observations"][2].__setitem__("minute", 1)
        )
        self.assert_spec_error(payload, "workloads[0].observations[2].minute")

    def test_negative_request_rejected(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["observations"][0].__setitem__("requests", -1)
        )
        self.assert_spec_error(payload, "workloads[0].observations[0].requests")

    def test_errors_must_not_exceed_requests(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][1]["observations"][2].__setitem__("errors", 101)
        )
        self.assert_spec_error(payload, "workloads[1].observations[2].errors")

    def test_min_must_not_exceed_max(self) -> None:
        payload = mutated_sample(lambda data: data["workloads"][0].__setitem__("min_replicas", 5))
        self.assert_spec_error(payload, "workloads[0].min_replicas")

    def test_cooldown_must_be_positive(self) -> None:
        payload = mutated_sample(lambda data: data["workloads"][0].__setitem__("cooldown_ms", 0))
        self.assert_spec_error(payload, "workloads[0].cooldown_ms")

    def test_budget_must_cover_minimum_replicas(self) -> None:
        payload = mutated_sample(lambda data: data.__setitem__("global_replica_budget", 2))
        self.assert_spec_error(payload, "global_replica_budget")

    def test_metric_is_restricted(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["rules"][0].__setitem__("metric", "cpu")
        )
        self.assert_spec_error(payload, "workloads[0].rules[0].metric")

    def test_target_percent_is_restricted(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["rules"][0].__setitem__("target_value", 101)
        )
        self.assert_spec_error(payload, "workloads[0].rules[0].target_value")

    def test_workloads_must_share_minute_grid(self) -> None:
        def shift_grid(data):
            for observation in data["workloads"][2]["observations"]:
                observation["minute"] += 1

        payload = mutated_sample(shift_grid)
        self.assert_spec_error(payload, "workloads[2].observations[0].minute")

    def test_bool_is_not_an_integer(self) -> None:
        payload = mutated_sample(
            lambda data: data["workloads"][0]["observations"][0].__setitem__("requests", True)
        )
        self.assert_spec_error(payload, "workloads[0].observations[0].requests")

    def test_invalid_json_is_rejected_as_whole_with_espec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"workloads": [}', encoding="utf-8")
            with self.assertRaises(BusinessError) as context:
                load_spec(path)
            self.assertEqual(context.exception.code, E_SPEC)
            self.assertEqual(context.exception.field_path, "$")

    def test_rules_are_order_independent_and_unique(self) -> None:
        payload = sample_payload()
        first_rule = payload["workloads"][0]["rules"][0]
        payload["workloads"][0]["rules"].append(json.loads(json.dumps(first_rule)))
        self.assert_spec_error(payload, "workloads[0].rules[1].id")


if __name__ == "__main__":
    unittest.main()
