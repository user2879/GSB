from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from config_arbiter.errors import SpecError
from config_arbiter.spec import load_spec

from .util import sample_object, write_temp_spec


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)
        self.payload = copy.deepcopy(sample_object())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def assert_spec_error(self, expected_field: str, payload: dict | None = None) -> SpecError:
        path = write_temp_spec(self.directory, payload or self.payload)
        with self.assertRaises(SpecError) as caught:
            load_spec(path)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, expected_field)
        return caught.exception

    def test_valid_sample_loads(self) -> None:
        path = write_temp_spec(self.directory, self.payload)
        spec = load_spec(path)
        self.assertEqual(set(spec.environments), {"cn-prod", "us-prod", "eu-prod"})
        self.assertEqual(set(spec.views), {"global-release", "us-eu-cpu"})
        self.assertEqual(spec.environments["cn-prod"].current_slice.version, 2)

    def test_duplicate_environment_id(self) -> None:
        self.payload["environments"][1]["id"] = "cn-prod"
        self.assert_spec_error("$.environments[1].id")

    def test_nonliteral_loopback_host(self) -> None:
        self.payload["environments"][0]["host"] = "localhost"
        self.assert_spec_error("$.environments[0].host")

    def test_duplicate_global_port(self) -> None:
        self.payload["environments"][1]["port"] = self.payload["environments"][0]["port"]
        self.assert_spec_error("$.environments[1].port")

    def test_slice_versions_must_strictly_increase(self) -> None:
        self.payload["environments"][0]["slices"][1]["version"] = 1
        self.assert_spec_error("$.environments[0].slices[1].version")

    def test_unknown_participant_is_located(self) -> None:
        self.payload["views"][0]["environments"][0] = "missing-env"
        self.assert_spec_error("$.views[0].environments[0]")

    def test_negative_tolerance_is_located(self) -> None:
        self.payload["views"][0]["keys"]["memory_limit"]["tolerance"] = -1
        self.assert_spec_error("$.views[0].keys.memory_limit.tolerance")

    def test_duplicate_view_id(self) -> None:
        self.payload["views"][1]["id"] = "global-release"
        self.assert_spec_error("$.views[1].id")

    def test_weight_must_survive_fixed_point_quantization(self) -> None:
        self.payload["environments"][0]["weight"] = "0.0001"
        self.assert_spec_error("$.environments[0].weight")

    def test_first_illegal_field_is_reported(self) -> None:
        self.payload["environments"][0]["host"] = "0.0.0.0"
        self.payload["views"][0]["keys"]["memory_limit"]["tolerance"] = -1
        self.assert_spec_error("$.environments[0].host")


if __name__ == "__main__":
    unittest.main()

