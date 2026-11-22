from __future__ import annotations

import copy
import json
import unittest
from meshwork.errors import MeshError
from meshwork.model import parse_requests, parse_spec
from tests.helpers import ROOT


class SpecValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads((ROOT / "samples" / "sample_spec.json").read_text(encoding="utf-8"))

    def assert_spec_error(self, raw: dict, location: str) -> MeshError:
        with self.assertRaises(MeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location, location)
        self.assertTrue(ctx.exception.field)
        return ctx.exception

    def test_sample_spec_is_valid_and_versions_are_stable(self) -> None:
        first = parse_spec(copy.deepcopy(self.raw))
        second = parse_spec(copy.deepcopy(self.raw))
        self.assertEqual(first.instance_version(), second.instance_version())
        self.assertEqual(first.config_digest(), second.config_digest())
        self.assertEqual([item.revision for item in first.route_revisions], [1, 2])

    def test_duplicate_service_id(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["services"][1]["id"] = raw["services"][0]["id"]
        self.assert_spec_error(raw, "$.services[1].id")

    def test_host_must_be_literal_loopback(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["services"][0]["instances"][0]["host"] = "localhost"
        self.assert_spec_error(raw, "$.services[0].instances[0].host")

    def test_ports_are_globally_unique(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["services"][0]["instances"][1]["port"] = raw["services"][0]["instances"][0]["port"]
        self.assert_spec_error(raw, "$.services[0].instances[1].port")

    def test_weight_must_be_nonnegative_integer(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["services"][0]["instances"][0]["weight"] = -1
        self.assert_spec_error(raw, "$.services[0].instances[0].weight")
        raw = copy.deepcopy(self.raw)
        raw["services"][0]["instances"][0]["weight"] = True
        self.assert_spec_error(raw, "$.services[0].instances[0].weight")

    def test_revisions_must_strictly_increase(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["route_revisions"][1]["revision"] = 1
        self.assert_spec_error(raw, "$.route_revisions[1].revision")

    def test_retry_backoff_elements_must_be_nonnegative_integers(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["session"]["retry_backoff_ms"][1] = -3
        self.assert_spec_error(raw, "$.session.retry_backoff_ms[1]")

    def test_behavior_percentages_must_sum_to_100(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["services"][0]["instances"][0]["behavior"]["error_percent"] = 6
        self.assert_spec_error(raw, "$.services[0].instances[0].behavior")

    def test_duplicate_request_id(self) -> None:
        raw = {
            "requests": [
                {"id": "r1", "service": "checkout"},
                {"id": "r1", "service": "checkout"},
            ]
        }
        with self.assertRaises(MeshError) as ctx:
            parse_requests(raw, ["checkout"])
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location, "$.requests[1].id")


if __name__ == "__main__":
    unittest.main()
