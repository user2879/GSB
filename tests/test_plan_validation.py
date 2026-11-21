from __future__ import annotations

import copy
import unittest

from memdiag.errors import DependencyError, PlanError, SampleError
from memdiag.plan import apply_plan_changes, plan_from_dict, sample_set_from_dict
from memdiag.util import generate_sample_payload
from tests.helpers import fixture_plan, tiny_plan_dict


class PlanValidationTests(unittest.TestCase):
    def test_supported_plan_loads_with_canonical_layer_order(self):
        plan = fixture_plan()
        self.assertEqual(plan.layer_ids, tuple(sorted(plan.layer_ids)))
        self.assertEqual(plan.plan_version, 1)
        self.assertGreater(plan.capacity_blocks, 0)

    def test_first_invalid_layer_is_reported_deterministically(self):
        document = tiny_plan_dict()
        document["layers"].extend(
            [
                {
                    "id": "z_bad",
                    "input_dim": 4,
                    "output_dim": 4,
                    "param_count": 10,
                    "precision": "float32",
                    "scale": 7,
                },
                {
                    "id": "a_bad",
                    "input_dim": 4,
                    "output_dim": 4,
                    "param_count": 10,
                    "precision": "int8",
                    "scale": 1,
                },
            ]
        )
        with self.assertRaises(PlanError) as caught:
            plan_from_dict(document)
        self.assertEqual(caught.exception.code, "E_PLAN")
        self.assertEqual(caught.exception.details["first_invalid_layer"], "a_bad")
        self.assertEqual(caught.exception.details["field"], "precision")

    def test_duplicate_layer_id_is_E_PLAN(self):
        document = tiny_plan_dict()
        document["layers"].append(copy.deepcopy(document["layers"][0]))
        with self.assertRaises(PlanError) as caught:
            plan_from_dict(document)
        self.assertEqual(caught.exception.code, "E_PLAN")
        self.assertEqual(caught.exception.details["layer_id"], "a")
        self.assertEqual(caught.exception.details["field"], "id")

    def test_positive_dimensions_and_scale_range(self):
        for field, value in [
            ("input_dim", 0),
            ("output_dim", -1),
            ("param_count", 0),
            ("scale", -1),
            ("scale", 7),
        ]:
            document = tiny_plan_dict()
            document["layers"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(PlanError):
                plan_from_dict(document)

    def test_dangling_reference_lists_referrer(self):
        document = tiny_plan_dict()
        document["consumers"]["a"] = ["b", "ghost"]
        with self.assertRaises(DependencyError) as caught:
            plan_from_dict(document)
        self.assertEqual(caught.exception.code, "E_DEPEND")
        self.assertEqual(caught.exception.details["referrers"], ["a"])
        self.assertEqual(caught.exception.details["missing_targets"], ["ghost"])

    def test_cycle_is_E_DEPEND_with_cycle(self):
        document = tiny_plan_dict()
        document["consumers"]["c"] = ["a"]
        with self.assertRaises(DependencyError) as caught:
            plan_from_dict(document)
        self.assertEqual(caught.exception.code, "E_DEPEND")
        self.assertTrue(caught.exception.details["cycle"])
        self.assertEqual(caught.exception.details["cycle"][0], caught.exception.details["cycle"][-1])

    def test_delete_referenced_layer_lists_all_referrers(self):
        plan = fixture_plan()
        with self.assertRaises(DependencyError) as caught:
            apply_plan_changes(plan, {"delete_layers": ["branch_a"]})
        self.assertEqual(caught.exception.code, "E_DEPEND")
        self.assertEqual(caught.exception.details["referrers"], ["merge", "stem"])

    def test_delete_unreferenced_sink_succeeds(self):
        plan = fixture_plan()
        new_plan, changed, _ = apply_plan_changes(
            plan,
            {
                "consumer_updates": {"merge": []},
                "delete_layers": ["head"],
            },
        )
        self.assertNotIn("head", new_plan.layer_ids)
        self.assertEqual(changed["head"], "layer_deleted")

    def test_generated_samples_are_deterministic(self):
        first = generate_sample_payload(seed=42, count=5, dim=3)
        second = generate_sample_payload(seed=42, count=5, dim=3)
        third = generate_sample_payload(seed=43, count=5, dim=3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        parsed = sample_set_from_dict(first)
        self.assertEqual(len(parsed.samples), 5)
        self.assertEqual(parsed.dim, 3)

    def test_invalid_sample_payload_rejected(self):
        document = generate_sample_payload(seed=42, count=2, dim=3)
        document["samples"][0]["values"] = [1.0, 2.0]
        with self.assertRaises(SampleError):
            sample_set_from_dict(document)


if __name__ == "__main__":
    unittest.main()
