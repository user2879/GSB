from __future__ import annotations

import unittest

from mixedmem.errors import DiagnosticError
from mixedmem.graph import build_graph
from mixedmem.samples import SampleFactory
from mixedmem.schema import load_plan, parse_plan, remove_layer

from .helpers import PLAN_PATH, plan_copy


class SchemaTests(unittest.TestCase):
    def test_sample_plan_is_valid(self) -> None:
        plan = load_plan(PLAN_PATH)
        self.assertEqual([layer.id for layer in plan.layers], ["embed", "hidden", "compress", "head"])
        self.assertEqual(plan.version, 1)
        self.assertEqual(build_graph(plan).order, ("embed", "hidden", "compress", "head"))

    def test_duplicate_layer_rejected_with_first_field(self) -> None:
        raw = plan_copy()
        raw["layers"][1]["id"] = "embed"
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        error = context.exception
        self.assertEqual(error.code, "E_SCHEMA")
        self.assertEqual(error.layer_id, "embed")
        self.assertEqual(error.field, "id")

    def test_bad_scale_points_at_layer_and_field(self) -> None:
        raw = plan_copy()
        raw["layers"][2]["scale"] = 999
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        error = context.exception
        self.assertEqual(error.code, "E_SCHEMA")
        self.assertEqual(error.layer_id, "compress")
        self.assertEqual(error.field, "scale")

    def test_missing_reference_is_rejected(self) -> None:
        raw = plan_copy()
        raw["layers"][1]["depends_on"] = ["does-not-exist"]
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        self.assertEqual(context.exception.code, "E_SCHEMA")
        self.assertEqual(context.exception.layer_id, "hidden")
        self.assertEqual(context.exception.field, "depends_on[0]")

    def test_dependency_cycle_is_rejected(self) -> None:
        raw = plan_copy()
        raw["layers"][0]["depends_on"] = ["head"]
        raw["layers"][0]["input_dim"] = 3
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        self.assertEqual(context.exception.code, "E_SCHEMA")
        self.assertIn("cycle", context.exception.message)
        self.assertEqual(context.exception.field, "dependency_graph")

    def test_self_memory_derivation_is_rejected(self) -> None:
        raw = plan_copy()
        raw["layers"][0]["memory_derivation"]["terms"] = [
            {"layer": "embed", "coefficient": 1}
        ]
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        self.assertEqual(context.exception.code, "E_SCHEMA")
        self.assertEqual(context.exception.field, "memory_derivation.terms[0].layer")

    def test_invalid_ttl_is_rejected(self) -> None:
        raw = plan_copy()
        raw["lease_ttl_ms"] = 0
        with self.assertRaises(DiagnosticError) as context:
            parse_plan(raw)
        self.assertEqual(context.exception.code, "E_SCHEMA")
        self.assertEqual(context.exception.field, "lease_ttl_ms")

    def test_referenced_deletion_is_e_depend_with_referrers(self) -> None:
        plan = parse_plan(plan_copy())
        with self.assertRaises(DiagnosticError) as context:
            remove_layer(plan, "hidden")
        error = context.exception
        self.assertEqual(error.code, "E_DEPEND")
        self.assertIn("compress", error.details["referrers"])
        self.assertIn("head", error.details["referrers"])

    def test_unreferenced_deletion_bumps_version(self) -> None:
        plan = parse_plan(plan_copy())
        revised = remove_layer(plan, "head")
        self.assertNotIn("head", revised.layer_by_id)
        self.assertEqual(revised.version, 2)

    def test_samples_are_bitwise_deterministic(self) -> None:
        plan = parse_plan(plan_copy())
        left = SampleFactory(plan, 12).summary()
        right = SampleFactory(plan, 12).summary()
        self.assertEqual(left, right)
        self.assertEqual(left["dataset_digest"], right["dataset_digest"])


if __name__ == "__main__":
    unittest.main()
