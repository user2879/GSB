from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.errors import E_SPEC, SpecError
from affinity_scheduler.samples import sample_spec
from affinity_scheduler.spec import load_spec

from tests.helpers import write_json


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _load(self, value: dict) -> None:
        path = self.root / "spec.json"
        write_json(path, value)
        load_spec(path)

    def _assert_spec_error(self, value: dict, field_path: str) -> SpecError:
        path = self.root / "spec.json"
        write_json(path, value)
        with self.assertRaises(SpecError) as caught:
            load_spec(path)
        error = caught.exception
        self.assertEqual(error.code, E_SPEC)
        self.assertEqual(error.field_path, field_path)
        self.assertIsNotNone(error.source_line)
        self.assertGreaterEqual(error.source_line or 0, 1)
        return error

    def test_valid_sample(self) -> None:
        path = self.root / "spec.json"
        write_json(path, sample_spec(1, 40100))
        spec = load_spec(path)
        self.assertEqual(spec.parameter_version, 1)
        self.assertEqual(spec.scale_factor, 1000)
        self.assertEqual(sum(spec.weights_myriad.values()), 10000)
        self.assertEqual(len(spec.nodes), 4)

    def test_duplicate_node_name_has_path_and_line(self) -> None:
        value = sample_spec(1, 40100)
        value["nodes"][1]["name"] = value["nodes"][0]["name"]
        error = self._assert_spec_error(value, "nodes[1].name")
        self.assertGreater(error.source_line or 0, 1)

    def test_duplicate_pod_id(self) -> None:
        value = sample_spec(1, 40100)
        value["pods"][1]["id"] = value["pods"][0]["id"]
        self._assert_spec_error(value, "pods[1].id")

    def test_nonpositive_resources(self) -> None:
        value = sample_spec(1, 40100)
        value["nodes"][0]["allocatable"]["cpu"] = 0
        self._assert_spec_error(value, "nodes[0].allocatable.cpu")
        value = sample_spec(1, 40100)
        value["pods"][0]["requests"]["memory"] = -1
        self._assert_spec_error(value, "pods[0].requests.memory")

    def test_taint_effect_must_be_supported_integer(self) -> None:
        value = sample_spec(1, 40100)
        value["nodes"][1]["taints"][0]["effect"] = 9
        self._assert_spec_error(value, "nodes[1].taints[0].effect")
        value = sample_spec(1, 40100)
        value["nodes"][1]["taints"][0]["effect"] = "NoSchedule"
        self._assert_spec_error(value, "nodes[1].taints[0].effect")

    def test_weight_sum_exactly_ten_thousand(self) -> None:
        value = sample_spec(1, 40100)
        value["plugins"]["node_affinity"]["weight_myriad"] = 3400
        self._assert_spec_error(value, "plugins")

    def test_scale_exponent_range_and_integer(self) -> None:
        value = sample_spec(1, 40100)
        value["scale_exponent"] = 4
        self._assert_spec_error(value, "scale_exponent")
        value = sample_spec(1, 40100)
        value["scale_exponent"] = 1.5
        self._assert_spec_error(value, "scale_exponent")

    def test_loopback_only_service(self) -> None:
        value = sample_spec(1, 40100)
        value["service"]["host"] = "0.0.0.0"
        self._assert_spec_error(value, "service.host")


if __name__ == "__main__":
    unittest.main()
