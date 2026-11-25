import unittest

from arbplatform.errors import SpecError
from arbplatform.spec import parse_manifest


def base_manifest():
    return {
        "scale_power": 2,
        "global_concurrency": 4,
        "environments": [
            {"env_id": "e1", "host": "127.0.0.1", "port": 49001,
             "lease_capacity": 2, "seed": 1, "weight": 1},
            {"env_id": "e2", "host": "127.0.0.1", "port": 49002,
             "lease_capacity": 1, "seed": 2, "weight": 2},
        ],
        "instances": [
            {"instance_id": "i1", "env_ref": "e1", "seq_low": 1,
             "seq_high": 3, "time_base_ms": 10},
            {"instance_id": "i2", "env_ref": "e2", "seq_low": 1,
             "seq_high": 2, "time_base_ms": 20},
        ],
        "views": [
            {"view_id": "v1", "env_refs": ["e1", "e2"],
             "policies": {"k": "latest_revision"}, "tolerance": 0}
        ],
    }


class SpecValidationTest(unittest.TestCase):
    def test_valid_manifest(self):
        manifest = parse_manifest(base_manifest())
        self.assertEqual(manifest.scale_power, 2)
        self.assertEqual(manifest.views[0].env_refs, ("e1", "e2"))

    def test_duplicate_env_id(self):
        data = base_manifest()
        data["environments"][1]["env_id"] = "e1"
        data["environments"][1]["port"] = 49003
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("environments[1].env_id", ctx.exception.field)

    def test_duplicate_port(self):
        data = base_manifest()
        data["environments"][1]["port"] = 49001
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("port", ctx.exception.field)

    def test_non_loopback_host(self):
        data = base_manifest()
        data["environments"][0]["host"] = "10.0.0.1"
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertEqual(ctx.exception.field, "$.environments[0].host")

    def test_non_positive_lease(self):
        data = base_manifest()
        data["environments"][0]["lease_capacity"] = 0
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertIn("lease_capacity", ctx.exception.field)

    def test_boolean_rejected_as_integer(self):
        data = base_manifest()
        data["environments"][0]["lease_capacity"] = True
        with self.assertRaises(SpecError):
            parse_manifest(data)

    def test_missing_env_reference(self):
        data = base_manifest()
        data["views"][0]["env_refs"] = ["e1", "eX"]
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertIn("env_refs", ctx.exception.field)

    def test_negative_tolerance(self):
        data = base_manifest()
        data["views"][0]["tolerance"] = -1
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertIn("tolerance", ctx.exception.field)

    def test_bad_policy(self):
        data = base_manifest()
        data["views"][0]["policies"]["k"] = "no-such-policy"
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertIn("policies.k", ctx.exception.field)

    def test_bad_scale_power(self):
        data = base_manifest()
        data["scale_power"] = 4
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertEqual(ctx.exception.field, "$.scale_power")

    def test_first_invalid_field_is_reported(self):
        data = base_manifest()
        data["environments"][0]["port"] = 0
        data["environments"][1]["lease_capacity"] = -3
        with self.assertRaises(SpecError) as ctx:
            parse_manifest(data)
        self.assertEqual(ctx.exception.field, "$.environments[0].port")


if __name__ == "__main__":
    unittest.main()
