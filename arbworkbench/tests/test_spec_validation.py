import copy
import unittest

from arbworkbench.errors import SpecError
from arbworkbench.spec import load_json, parse_specification

from tests import SAMPLE_ENV, SAMPLE_VIEWS


class SpecValidationTests(unittest.TestCase):
    def setUp(self):
        self.env_doc = load_json(SAMPLE_ENV)
        self.view_doc = load_json(SAMPLE_VIEWS)

    def test_sample_loads(self):
        spec = parse_specification(self.env_doc, self.view_doc)
        self.assertEqual(spec.env_ids,
                         ("env-canary", "env-dr", "env-prod-eu",
                          "env-prod-us", "env-staging"))
        self.assertEqual(len(spec.views), 2)

    def test_duplicate_environment_id(self):
        doc = copy.deepcopy(self.env_doc)
        doc["environments"][1]["env_id"] = doc["environments"][0]["env_id"]
        with self.assertRaises(SpecError) as ctx:
            parse_specification(doc, self.view_doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("env_id", ctx.exception.location)

    def test_duplicate_observation_seq(self):
        doc = copy.deepcopy(self.env_doc)
        doc["environments"][0]["observations"]["timeout_ms"][1]["seq"] = 1
        with self.assertRaises(SpecError) as ctx:
            parse_specification(doc, self.view_doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("seq", ctx.exception.location)

    def test_negative_timestamp(self):
        doc = copy.deepcopy(self.env_doc)
        doc["environments"][0]["observations"]["timeout_ms"][0]["timestamp_ms"] = -1
        with self.assertRaises(SpecError) as ctx:
            parse_specification(doc, self.view_doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("timestamp_ms", ctx.exception.location)

    def test_unknown_env_reference(self):
        doc = copy.deepcopy(self.view_doc)
        doc["views"][0]["env_ids"].append("env-ghost")
        with self.assertRaises(SpecError) as ctx:
            parse_specification(self.env_doc, doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("env-ghost", ctx.exception.location)

    def test_unsupported_policy(self):
        doc = copy.deepcopy(self.view_doc)
        doc["views"][0]["policies"]["timeout_ms"] = "highest_priority_override"
        with self.assertRaises(SpecError) as ctx:
            parse_specification(self.env_doc, doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("policies", ctx.exception.location)

    def test_malformed_json_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(SpecError) as ctx:
                load_json(bad)
            self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_bad_interval_shape(self):
        doc = copy.deepcopy(self.env_doc)
        doc["environments"][0]["observations"]["maintenance_window"][0]["value"] = [1]
        with self.assertRaises(SpecError) as ctx:
            parse_specification(doc, self.view_doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")


if __name__ == "__main__":
    unittest.main()
