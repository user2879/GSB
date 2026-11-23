from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest

from gateway_diag.model import RequestSample, parse_spec_obj
from gateway_diag.parameters import generate_parameters, parameter_change_sources
from gateway_diag.routing import plan_routes, stable_bucket

from tests.helpers import base_spec, sample_requests


class ParameterTests(unittest.TestCase):
    def test_parameter_digest_is_bit_reproducible(self) -> None:
        spec = parse_spec_obj(base_spec([44041, 44042]))
        first = [generate_parameters(backend) for backend in spec.backends]
        second = [generate_parameters(backend) for backend in spec.backends]
        self.assertEqual(
            [item.parameter_digest for item in first],
            [item.parameter_digest for item in second],
        )
        self.assertEqual(first[0].weight_in.tobytes(), second[0].weight_in.tobytes())

    def test_digest_is_reproducible_in_a_separate_process(self) -> None:
        raw = base_spec([44051, 44052])
        script = r"""
import json, sys
from gateway_diag.model import parse_spec_obj
from gateway_diag.parameters import generate_parameters
spec = parse_spec_obj(json.loads(sys.stdin.read()))
print(json.dumps([generate_parameters(b).parameter_digest for b in spec.backends]))
"""
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "random"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(raw),
            text=True,
            capture_output=True,
            check=True,
            env=environment,
        )
        remote = json.loads(completed.stdout)
        local = [generate_parameters(backend).parameter_digest for backend in parse_spec_obj(raw).backends]
        self.assertEqual(remote, local)

    def test_digest_change_sources_are_localized(self) -> None:
        raw = base_spec([44061, 44062])
        base = parse_spec_obj(raw)
        original = generate_parameters(base.backends[0])
        changed_raw = copy.deepcopy(raw)
        changed_raw["backends"][0]["model_dim"] = 5
        changed_raw["backends"][0]["seed"] = 999
        changed = generate_parameters(parse_spec_obj(changed_raw).backends[0])
        self.assertNotEqual(original.parameter_digest, changed.parameter_digest)
        self.assertEqual(parameter_change_sources(original, changed), ["model_dim", "seed"])

        quant_raw = copy.deepcopy(raw)
        quant_raw["backends"][0]["quantization"] = "fp32"
        quantized = generate_parameters(parse_spec_obj(quant_raw).backends[0])
        self.assertIn("quantization", parameter_change_sources(original, quantized))
        self.assertIn("quantization_version", parameter_change_sources(original, quantized))
        self.assertNotEqual(original.parameter_digest, quantized.parameter_digest)

    def test_breakdown_binds_identity_dimension_and_quantization(self) -> None:
        spec = parse_spec_obj(base_spec([44071, 44072]))
        breakdown = generate_parameters(spec.backends[0]).breakdown()
        self.assertEqual(breakdown["backend_id"], "stable")
        self.assertEqual(breakdown["model_dim"], 4)
        self.assertEqual(breakdown["quantization_version"], "symmetric-int8-v1")
        self.assertEqual(breakdown["parameter_digest"], breakdown["parameter_digest"].lower())
        self.assertEqual(len(breakdown["parameter_digest"]), 64)


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = parse_spec_obj(base_spec([44081, 44082]))
        self.requests = [
            RequestSample(
                request_id=value["request_id"],
                client_key=value["client_key"],
                input_vector=tuple(value["input"]),
                arrival_ms=value["arrival_ms"],
                source_index=index,
            )
            for index, value in enumerate(sample_requests(20))
        ]

    def test_bucket_range_and_stability(self) -> None:
        for request in self.requests:
            first = stable_bucket("alpha", request.client_key)
            second = stable_bucket("alpha", request.client_key)
            self.assertEqual(first, second)
            self.assertGreaterEqual(first, 0)
            self.assertLess(first, 10000)

    def test_weight_boundaries(self) -> None:
        raw = base_spec([44091, 44092])
        raw["rules"][0]["canary_weight_bp"] = 0
        stable_spec = parse_spec_obj(raw)
        stable_plan = plan_routes(stable_spec, self.requests)
        self.assertTrue(all(item.target == "stable" for item in stable_plan.decisions))

        raw["rules"][0]["canary_weight_bp"] = 10000
        canary_spec = parse_spec_obj(raw)
        canary_plan = plan_routes(canary_spec, self.requests)
        self.assertTrue(all(item.target == "canary" for item in canary_plan.decisions))

    def test_plan_does_not_depend_on_input_order(self) -> None:
        first = plan_routes(self.spec, self.requests)
        second = plan_routes(self.spec, list(reversed(self.requests)))
        self.assertEqual(
            [decision.to_dict() for decision in first.decisions],
            [decision.to_dict() for decision in second.decisions],
        )
        self.assertEqual(
            [batch.to_dict() for batch in first.batches],
            [batch.to_dict() for batch in second.batches],
        )

    def test_same_client_stays_on_same_side_for_same_rule_and_weight(self) -> None:
        first = plan_routes(self.spec, self.requests)
        second = plan_routes(self.spec, self.requests)
        self.assertEqual(
            [(item.request_id, item.target) for item in first.decisions],
            [(item.request_id, item.target) for item in second.decisions],
        )


if __name__ == "__main__":
    unittest.main()
