from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config_arbiter.errors import RefError, SpecError
from config_arbiter.spec import load_specs
from tests.support import copy_samples, read_json, write_json


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment_path, self.views_path = copy_samples(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sample_loads(self) -> None:
        specs = load_specs(self.environment_path, self.views_path)
        self.assertEqual(sorted(specs.environments), ["dr-ap", "prod-cn", "staging-eu"])
        self.assertEqual(sorted(specs.views), ["cost-focus", "ops-authority", "release-guard"])
        self.assertEqual(specs.session.scale_power, 3)

    def test_negative_depth_is_rejected_as_spec(self) -> None:
        document = read_json(self.environment_path)
        document["environments"][0]["depth"] = -1
        write_json(self.environment_path, document)
        with self.assertRaises(SpecError) as raised:
            load_specs(self.environment_path, self.views_path)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.details["path"], "$.environments[0].depth")

    def test_duplicate_environment_id_is_rejected(self) -> None:
        document = read_json(self.environment_path)
        document["environments"][1]["environment_id"] = document["environments"][0]["environment_id"]
        write_json(self.environment_path, document)
        with self.assertRaises(SpecError) as raised:
            load_specs(self.environment_path, self.views_path)
        self.assertIn("duplicate environment_id", raised.exception.message)
        self.assertEqual(raised.exception.details["path"], "$.environments[1].environment_id")

    def test_duplicate_key_in_same_source_is_rejected(self) -> None:
        document = read_json(self.environment_path)
        observation = document["environments"][0]["sources"][0]["observations"][0]
        document["environments"][0]["sources"][0]["observations"].append(dict(observation))
        write_json(self.environment_path, document)
        with self.assertRaises(SpecError) as raised:
            load_specs(self.environment_path, self.views_path)
        self.assertIn("duplicate key", raised.exception.message)
        self.assertEqual(raised.exception.details["entry"], "environments[0].sources[0].observations[6]")

    def test_boolean_value_is_not_a_supported_number(self) -> None:
        document = read_json(self.environment_path)
        document["environments"][0]["sources"][0]["observations"][0]["value"] = True
        write_json(self.environment_path, document)
        with self.assertRaises(SpecError) as raised:
            load_specs(self.environment_path, self.views_path)
        self.assertIn("finite JSON integer or float", raised.exception.message)

    def test_missing_view_environment_is_spec_error(self) -> None:
        document = read_json(self.views_path)
        document["views"][0]["environments"].append("missing-env")
        write_json(self.views_path, document)
        with self.assertRaises(SpecError) as raised:
            load_specs(self.environment_path, self.views_path)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertIn("missing environment", raised.exception.message)

    def test_priority_reference_missing_environment_is_ref_error(self) -> None:
        with self.assertRaises(RefError) as raised:
            load_specs(
                self.environment_path,
                self.views_path,
                inject_priority_ref_missing_env=True,
            )
        self.assertEqual(raised.exception.code, "E_REF")
        self.assertEqual(
            raised.exception.details["missing_environment_id"],
            "__injected_missing_environment__",
        )

    def test_spec_corruption_injection_returns_first_entry(self) -> None:
        with self.assertRaises(SpecError) as raised:
            load_specs(
                self.environment_path,
                self.views_path,
                inject_spec_corrupt=True,
            )
        self.assertEqual(raised.exception.details["entry"], "environments[0]")
        self.assertEqual(raised.exception.details["path"], "$.environments[0].depth")


if __name__ == "__main__":
    unittest.main()
