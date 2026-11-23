from __future__ import annotations

import copy
import json
import unittest

from gateway_diag.errors import SpecError
from gateway_diag.model import parse_request_objs, parse_spec_obj

from tests.helpers import base_spec


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = base_spec([44011, 44012])

    def assert_spec_error(self, value, entry: str, field: str) -> None:
        with self.assertRaises(SpecError) as caught:
            parse_spec_obj(value)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.entry, entry)
        self.assertEqual(caught.exception.field, field)

    def test_valid_spec_and_canonical_fingerprint(self) -> None:
        first = parse_spec_obj(self.raw)
        second = parse_spec_obj(copy.deepcopy(self.raw))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.backends), 2)
        self.assertEqual(first.rules[0].version, 1)

    def test_duplicate_backend_id_is_first_illegal_entry(self) -> None:
        self.raw["backends"][1]["id"] = "stable"
        self.assert_spec_error(self.raw, "backends[1]", "id")

    def test_duplicate_port_is_rejected_globally(self) -> None:
        self.raw["backends"][1]["port"] = self.raw["backends"][0]["port"]
        self.assert_spec_error(self.raw, "backends[1]", "port")

    def test_host_must_be_exact_loopback_literal(self) -> None:
        self.raw["backends"][0]["host"] = "0.0.0.0"
        self.assert_spec_error(self.raw, "backends[0]", "host")
        self.raw["backends"][0]["host"] = "localhost"
        self.assert_spec_error(self.raw, "backends[0]", "host")

    def test_dimensions_are_positive_integers_and_bool_is_not_int(self) -> None:
        for field, value in (
            ("model_dim", 0),
            ("model_dim", 3.5),
            ("model_dim", True),
            ("hidden_dim", -1),
        ):
            with self.subTest(field=field, value=value):
                raw = copy.deepcopy(self.raw)
                raw["backends"][0][field] = value
                self.assert_spec_error(raw, "backends[0]", field)

    def test_seed_and_quantization(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["backends"][0]["seed"] = "7"
        self.assert_spec_error(raw, "backends[0]", "seed")
        raw = copy.deepcopy(self.raw)
        raw["backends"][0]["quantization"] = "fp16"
        self.assert_spec_error(raw, "backends[0]", "quantization")

    def test_gray_weight_batch_size_and_targets(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["rules"][0]["canary_weight_bp"] = 10001
        self.assert_spec_error(raw, "rules[0]", "canary_weight_bp")
        raw = copy.deepcopy(self.raw)
        raw["rules"][0]["batch_size"] = 0
        self.assert_spec_error(raw, "rules[0]", "batch_size")
        raw = copy.deepcopy(self.raw)
        raw["rules"][0]["stable_backend"] = "missing"
        self.assert_spec_error(raw, "rules[0]", "stable_backend")
        raw = copy.deepcopy(self.raw)
        raw["rules"][0]["canary_backend"] = "missing"
        self.assert_spec_error(raw, "rules[0]", "canary_backend")

    def test_rule_id_uniqueness_and_version(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["rules"].append(copy.deepcopy(raw["rules"][0]))
        self.assert_spec_error(raw, "rules[1]", "id")
        raw = copy.deepcopy(self.raw)
        raw["rules"][0]["version"] = 0
        self.assert_spec_error(raw, "rules[0]", "version")

    def test_limits_validation(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["limits"]["global_concurrency"] = 0
        self.assert_spec_error(raw, "limits", "global_concurrency")
        raw = copy.deepcopy(self.raw)
        raw["limits"]["backend_concurrency"]["ghost"] = 1
        self.assert_spec_error(raw, "limits", "backend_concurrency.ghost")


class RequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = parse_spec_obj(base_spec([44021, 44022]))
        self.raw = [
            {
                "request_id": 1,
                "client_key": "alpha:one",
                "input": [0.1, 0.2, 0.3, 0.4],
                "arrival_ms": 1000,
            }
        ]

    def parse(self, values):
        return parse_request_objs(
            [(index + 1, json.dumps(value, ensure_ascii=False)) for index, value in enumerate(values)],
            self.spec,
        )

    def test_requests_are_ordered_by_arrival_then_id(self) -> None:
        values = [
            dict(self.raw[0], request_id=2, arrival_ms=1002, client_key="alpha:two"),
            dict(self.raw[0], request_id=1, arrival_ms=1001),
        ]
        requests = self.parse(values)
        self.assertEqual([request.request_id for request in requests], [1, 2])

    def test_duplicate_request_id(self) -> None:
        values = [self.raw[0], dict(self.raw[0])]
        with self.assertRaises(SpecError) as caught:
            self.parse(values)
        self.assertEqual(caught.exception.entry, "requests[2]")
        self.assertEqual(caught.exception.field, "request_id")

    def test_wrong_routed_dimension_is_rejected(self) -> None:
        values = [dict(self.raw[0], input=[0.1, 0.2])]
        with self.assertRaises(SpecError) as caught:
            self.parse(values)
        self.assertEqual(caught.exception.entry, "requests[1]")
        self.assertEqual(caught.exception.field, "input")

    def test_unknown_prefix_is_rejected(self) -> None:
        raw = base_spec([44031, 44032])
        raw["rules"].append(
            {
                "id": "beta",
                "prefix": "beta:",
                "stable_backend": "stable",
                "canary_backend": "canary",
                "canary_weight_bp": 0,
                "batch_size": 1,
                "version": 1,
            }
        )
        spec = parse_spec_obj(raw)
        with self.assertRaises(SpecError) as caught:
            parse_request_objs(
                [
                    (
                        1,
                        json.dumps(
                            {
                                "request_id": 1,
                                "client_key": "gamma:unknown",
                                "input": [0, 0, 0, 0],
                                "arrival_ms": 1,
                            }
                        ),
                    )
                ],
                spec,
            )
        self.assertEqual(caught.exception.field, "client_key")


if __name__ == "__main__":
    unittest.main()
