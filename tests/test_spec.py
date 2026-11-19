from __future__ import annotations

import copy
import unittest

import yaml

from story_workbench.errors import E_SPEC_INVALID, SpecInvalid
from story_workbench.samples import boundary_spec, injection_spec, merge_spec, normal_spec
from story_workbench.spec import parse_spec

try:
    from tests.helpers import parse_document
except ModuleNotFoundError:  # supports `unittest discover -s tests`
    from helpers import parse_document


class SpecValidationTests(unittest.TestCase):
    def test_all_fixed_specs_parse_and_serialize_deterministically(self):
        for factory in (normal_spec, merge_spec, boundary_spec, injection_spec):
            document = factory()
            text = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
            spec = parse_spec(text, source_name=f"{document['id']}.yaml")
            repeated = parse_spec(text, source_name=f"{document['id']}.yaml")
            self.assertEqual(spec.spec_hash, repeated.spec_hash)
            self.assertEqual(spec.canonical_json(), repeated.canonical_json())
            self.assertEqual(spec.canonical_json(), spec.canonical_json())

    def test_unknown_effect_target_is_spec_invalid_with_edge_location(self):
        document = copy.deepcopy(normal_spec())
        document["edges"][1]["effects"][0]["target"] = "honor"
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        error = caught.exception
        self.assertEqual(error.code, E_SPEC_INVALID)
        self.assertEqual(error.location["edge_index"], 1)
        self.assertEqual(error.location["edge_id"], "e_ally")
        self.assertEqual(error.location["effect_id"], "f_ally_trust")

    def test_unknown_expression_variable_is_spec_invalid(self):
        document = copy.deepcopy(normal_spec())
        document["edges"][0]["effects"][0]["expr"] = "honor + 0.1"
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertIn("unknown state variable", caught.exception.message)
        self.assertEqual(caught.exception.location["edge_index"], 0)

    def test_cycle_is_rejected_with_edge_indices(self):
        document = copy.deepcopy(normal_spec())
        document["edges"].append(
            {
                "id": "e_cycle",
                "from": "s_ally",
                "to": "s_start",
                "label": "cycle",
                "effects": [],
                "conditions": [],
            }
        )
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertIn("cycle", caught.exception.message)
        self.assertTrue(caught.exception.details["cycle_edge_indices"])

    def test_nonfinite_threshold_is_rejected(self):
        document = copy.deepcopy(normal_spec())
        document["edges"][1]["conditions"][0]["threshold"] = float("nan")
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertEqual(caught.exception.location["edge_index"], 1)
        self.assertEqual(caught.exception.location["condition_id"], "c_trust_ally")

    def test_nonfinite_initial_value_is_rejected(self):
        document = copy.deepcopy(normal_spec())
        document["initial_state"]["trust"] = float("inf")
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertEqual(caught.exception.location["component"], "trust")

    def test_division_operator_is_not_allowed(self):
        document = copy.deepcopy(normal_spec())
        document["edges"][0]["effects"][0]["expr"] = "trust / 2.0"
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertIn("not allowed", caught.exception.message)

    def test_duplicate_yaml_mapping_key_is_rejected(self):
        text = yaml.safe_dump(normal_spec(), allow_unicode=True, sort_keys=False)
        text = text.replace("seed: 20260101\n", "seed: 20260101\nseed: 20260102\n", 1)
        with self.assertRaises(SpecInvalid) as caught:
            parse_spec(text, source_name="duplicate.yaml")
        self.assertIn("duplicate YAML mapping key", caught.exception.message)

    def test_sample_with_broken_source_continuity_is_rejected(self):
        document = copy.deepcopy(normal_spec())
        document["samples"][0]["edges"] = ["e_ally"]
        with self.assertRaises(SpecInvalid) as caught:
            parse_document(document)
        self.assertEqual(caught.exception.details["expected_source"], "s_start")


if __name__ == "__main__":
    unittest.main()
