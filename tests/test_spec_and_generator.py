from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from streamagg.errors import BusinessError
from streamagg.generator import generate_session
from streamagg.spec import load_spec, parse_spec


ROOT = Path(__file__).resolve().parents[1]


def base_spec() -> dict:
    return json.loads((ROOT / "samples" / "scenario1_ordered.json").read_text(encoding="utf-8"))


class SpecValidationTests(unittest.TestCase):
    def assert_code(self, payload: dict, code: str, field: str) -> None:
        with self.assertRaises(BusinessError) as caught:
            parse_spec(payload)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.field, field)

    def test_duplicate_source_rejected_at_first_violating_field(self) -> None:
        payload = base_spec()
        payload["sources"].append(copy.deepcopy(payload["sources"][0]))
        self.assert_code(payload, "E_SPEC", "$.sources[1].source_id")

    def test_duplicate_rule_rejected(self) -> None:
        payload = base_spec()
        payload["rules"].append(copy.deepcopy(payload["rules"][0]))
        self.assert_code(payload, "E_SPEC", "$.rules[1].rule_id")

    def test_seed_must_be_positive_integer(self) -> None:
        payload = base_spec()
        payload["sources"][0]["seed"] = 0
        self.assert_code(payload, "E_SPEC", "$.sources[0].seed")

    def test_bool_is_not_an_integer(self) -> None:
        payload = base_spec()
        payload["sources"][0]["seed"] = True
        self.assert_code(payload, "E_SPEC", "$.sources[0].seed")

    def test_window_width_must_be_positive(self) -> None:
        payload = base_spec()
        payload["rules"][0]["width_ms"] = -1
        self.assert_code(payload, "E_SPEC", "$.rules[0].width_ms")

    def test_lateness_must_be_positive(self) -> None:
        payload = base_spec()
        payload["rules"][0]["lateness_ms"] = 0
        self.assert_code(payload, "E_SPEC", "$.rules[0].lateness_ms")

    def test_out_of_order_ratio_upper_bound(self) -> None:
        payload = base_spec()
        payload["sources"][0]["out_of_order_ratio_bp"] = 10001
        self.assert_code(payload, "E_SPEC", "$.sources[0].out_of_order_ratio_bp")

    def test_unsupported_aggregation_rejected(self) -> None:
        payload = base_spec()
        payload["rules"][0]["aggregation_functions"] = ["COUNT", "MEDIAN"]
        self.assert_code(payload, "E_SPEC", "$.rules[0].aggregation_functions[1]")

    def test_invalid_scale_uses_grid_code(self) -> None:
        payload = base_spec()
        payload["session"]["fixed_scale"] = 0
        self.assert_code(payload, "E_GRID", "$.session.fixed_scale")

    def test_negative_budget_rejected(self) -> None:
        payload = base_spec()
        payload["session"]["lateness_budget_count"] = -1
        self.assert_code(payload, "E_SPEC", "$.session.lateness_budget_count")


class DeterministicGenerationTests(unittest.TestCase):
    def test_same_seed_produces_identical_bytes_and_digest(self) -> None:
        spec = load_spec(ROOT / "samples" / "sample_spec.json")
        events_a, generations_a = generate_session(spec)
        events_b, generations_b = generate_session(spec)
        self.assertEqual([event.to_dict() for event in events_a], [event.to_dict() for event in events_b])
        source_id = spec.sources[0].source_id
        self.assertEqual(generations_a[source_id].digest, generations_b[source_id].digest)

    def test_out_of_order_ratio_is_respected(self) -> None:
        spec = load_spec(ROOT / "samples" / "sample_spec.json")
        events, generations = generate_session(spec)
        generation = generations[spec.sources[0].source_id]
        self.assertLessEqual(generation.actual_out_of_order_ratio_bp, spec.sources[0].out_of_order_ratio_bp)
        events_by_partition: dict[str, list[int]] = {}
        for event in events:
            events_by_partition.setdefault(event.partition_key, []).append(event.event_time_ms)
        self.assertTrue(any(values != sorted(values) for values in events_by_partition.values()))


if __name__ == "__main__":
    unittest.main()
