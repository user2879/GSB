import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from hotgray import determinism as det
from hotgray.errors import SpecError
from hotgray.hashing import route_decision, stable_bucket
from hotgray.samples import build_invalid_declaration, build_sample_declaration
from hotgray.spec import validate_declaration


class DeterminismTests(unittest.TestCase):
    def test_parameters_bit_identical(self):
        p1 = det.build_parameters(101, 8, 6)
        p2 = det.build_parameters(101, 8, 6)
        for key in p1:
            self.assertTrue(np.array_equal(p1[key], p2[key]))
        d1 = det.parameter_digest("b", "v1", 1, 8, 6, 101, p1)
        d2 = det.parameter_digest("b", "v1", 1, 8, 6, 101, p2)
        self.assertEqual(d1, d2)

    def test_hotswap_seed_changes_digest(self):
        p0 = det.build_parameters(202, 8, 6)
        seed1 = det.next_hotswap_seed(202, 1)
        p1 = det.build_parameters(seed1, 8, 6)
        d0 = det.parameter_digest("g", "v1", 1, 8, 6, 202, p0)
        d1 = det.parameter_digest("g", "v1", 1, 8, 6, seed1, p1)
        self.assertNotEqual(d0, d1)

    def test_input_digest_stable(self):
        v1 = det.build_input_vector("r", "s1", 8)
        v2 = det.build_input_vector("r", "s1", 8)
        self.assertEqual(det.input_digest("r", "s1", 8, v1),
                         det.input_digest("r", "s1", 8, v2))

    def test_forward_pure(self):
        params = det.build_parameters(7, 5, 4)
        x = np.vstack([det.build_input_vector("r", f"s{i}", 5)
                       for i in range(3)])
        out1 = det.forward(params, x)
        out2 = det.forward(params, x)
        np.testing.assert_array_equal(out1, out2)


class RoutingTests(unittest.TestCase):
    def test_bucket_range_and_stability(self):
        for i in range(100):
            bucket = stable_bucket("rule-chat", f"client-{i}")
            self.assertIn(bucket, range(10000))
            self.assertEqual(
                route_decision("rule-chat", f"client-{i}", 4000),
                route_decision("rule-chat", f"client-{i}", 4000),
            )

    def test_weight_boundaries(self):
        self.assertEqual(route_decision("r", "k", 0), "stable")
        self.assertEqual(route_decision("r", "k", 10000), "gray")


class SpecValidationTests(unittest.TestCase):
    def test_valid_sample(self):
        self.assertTrue(validate_declaration(build_sample_declaration()))

    def test_invalid_port_rejected_with_path(self):
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(build_invalid_declaration())
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("backends[1].port", ctx.exception.field)

    def test_weight_out_of_range(self):
        data = build_sample_declaration()
        data["rules"][0]["gray_weight_bp"] = 10001
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(data)
        self.assertEqual(ctx.exception.field,
                         "rules[0].gray_weight_bp")

    def test_missing_backend_reference(self):
        data = build_sample_declaration()
        data["rules"][0]["gray_backend"] = "ghost"
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(data)
        self.assertEqual(ctx.exception.field, "rules[0].gray_backend")

    def test_duplicate_port(self):
        data = build_sample_declaration()
        data["backends"][1]["port"] = data["backends"][0]["port"]
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(data)
        self.assertEqual(ctx.exception.field, "backends[1].port")


if __name__ == "__main__":
    unittest.main()
