import json
import unittest
from pathlib import Path

from chaoskit.errors import SpecError
from chaoskit.spec import load_spec, parse_spec

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_experiment.json"


def base_payload():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


class SpecValidationTest(unittest.TestCase):
    def test_valid_sample_loads(self):
        spec = load_spec(str(SAMPLE))
        self.assertEqual(len(spec.services), 2)
        self.assertEqual(len(spec.injections), 5)
        self.assertEqual(spec.budget.max_parallel, 1)

    def test_root_must_be_object(self):
        with self.assertRaises(SpecError) as ctx:
            parse_spec([1, 2, 3])
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("$", ctx.exception.paths)

    def test_port_range_and_uniqueness(self):
        payload = base_payload()
        payload["services"][0]["data_port"] = 70000
        payload["services"][1]["proxy_port"] = payload["services"][1]["data_port"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        paths = ctx.exception.paths
        self.assertTrue(any("data_port" in path for path in paths))
        self.assertTrue(any("proxy_port" in path for path in paths))

    def test_duplicate_service_id_and_unknown_target(self):
        payload = base_payload()
        payload["services"][1]["service_id"] = payload["services"][0]["service_id"]
        payload["injections"][0]["target_id"] = "ghost"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertTrue(any("service_id" in path for path in ctx.exception.paths))
        self.assertTrue(any("target_id" in path for path in ctx.exception.paths))

    def test_injection_type_and_params(self):
        payload = base_payload()
        payload["injections"][0]["type"] = "blackhole"
        payload["injections"][2]["params"]["error_bps"] = 12000
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertTrue(any(".type" in path for path in ctx.exception.paths))
        self.assertTrue(any("error_bps" in path for path in ctx.exception.paths))

    def test_budget_positive_and_cap(self):
        payload = base_payload()
        payload["budget"]["max_injections"] = 0
        payload["budget"]["max_parallel"] = 0
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertIn("$.budget.max_injections", ctx.exception.paths)
        self.assertIn("$.budget.max_parallel", ctx.exception.paths)

    def test_budget_cap_below_injection_count(self):
        payload = base_payload()
        payload["budget"]["max_injections"] = 1
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertIn("$.budget.max_injections", ctx.exception.paths)

    def test_bool_is_not_integer(self):
        payload = base_payload()
        payload["seed"] = True
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertIn("$.seed", ctx.exception.paths)

    def test_loopback_only(self):
        payload = base_payload()
        payload["services"][0]["host"] = "10.0.0.1"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertTrue(any("host" in path for path in ctx.exception.paths))

    def test_rejection_is_all_or_nothing(self):
        payload = base_payload()
        payload["seed"] = -1
        payload["injections"][0]["duration_ms"] = 0
        with self.assertRaises(SpecError) as ctx:
            parse_spec(payload)
        self.assertGreaterEqual(len(ctx.exception.paths), 2)


if __name__ == "__main__":
    unittest.main()
