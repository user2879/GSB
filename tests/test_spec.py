import json
import os
import tempfile
import unittest

from swa.errors import BusinessError
from swa.spec import load_sample_overrides, load_spec, parse_spec


def write_tmp(payload, *, raw=None):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    handle.write(raw if raw is not None else json.dumps(payload))
    handle.close()
    return handle.name


class SpecValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = {
            "sources": [{"source_id": "a", "seed": 1, "event_count": 4}],
            "rules": [
                {
                    "rule_id": "r",
                    "window_seconds": 2,
                    "group_key": "group_key",
                    "aggregations": ["count", "sum"],
                    "concurrency": 1,
                    "memory_budget": 4,
                }
            ],
        }

    def test_valid_spec_and_sample_load(self):
        path = write_tmp(self.valid)
        spec = load_spec(path)
        self.assertEqual(len(spec.sources), 1)
        self.assertEqual(spec.rules[0].version, 1)
        sample = write_tmp({"events": [{"source_id": "a", "source_seq": 1, "value": 9}]})
        overrides = load_sample_overrides(sample)
        self.assertEqual(overrides[("a", 1)]["value"], 9)

    def test_rule_version_defaults_to_one(self):
        spec = parse_spec(self.valid)
        self.assertEqual(spec.rules[0].version, 1)

    def test_duplicate_source_is_rejected_with_path(self):
        self.valid["sources"].append({"source_id": "a", "seed": 2, "event_count": 3})
        with self.assertRaises(BusinessError) as ctx:
            parse_spec(self.valid)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("sources[1].source_id", ctx.exception.details["field_errors"][0]["path"])

    def test_duplicate_rule_is_rejected_with_path(self):
        self.valid["rules"].append(dict(self.valid["rules"][0]))
        with self.assertRaises(BusinessError) as ctx:
            parse_spec(self.valid)
        self.assertTrue(any("rules[1].rule_id" in e["path"] for e in ctx.exception.details["field_errors"]))

    def test_invalid_aggregation_and_positive_fields(self):
        self.valid["rules"][0].update(
            {"window_seconds": 0, "aggregations": ["median"], "concurrency": 0, "memory_budget": -2}
        )
        with self.assertRaises(BusinessError) as ctx:
            parse_spec(self.valid)
        paths = {e["path"] for e in ctx.exception.details["field_errors"]}
        for expected in ("window_seconds", "aggregations", "concurrency", "memory_budget"):
            self.assertTrue(any(expected in p for p in paths), expected)

    def test_identity_and_count_aliases_are_supported(self):
        payload = {
            "sources": [{"id": "a", "seed": 1, "count": 4}],
            "rules": [
                {
                    "id": "r",
                    "window_seconds": 2,
                    "group_key": "group_key",
                    "aggregations": ["count"],
                    "concurrency": 1,
                    "memory_budget": 4,
                }
            ],
        }
        spec = parse_spec(payload)
        self.assertEqual(spec.sources[0].source_id, "a")
        self.assertEqual(spec.sources[0].event_count, 4)
        self.assertEqual(spec.rules[0].rule_id, "r")

    def test_ambiguous_identity_alias_is_rejected(self):
        self.valid["sources"][0]["id"] = "alias"
        # Both canonical source_id and alias id present: identity is ambiguous.
        self.valid["sources"][0]["source_id"] = "canonical"
        with self.assertRaises(BusinessError) as ctx:
            parse_spec(self.valid)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_duplicate_json_key_and_malformed_json_are_e_spec(self):
        raw = '{"sources": [{"source_id": "a", "seed": 1, "seed": 2, "event_count": 3}], "rules": []}'
        path = write_tmp(None, raw=raw)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        malformed = write_tmp(None, raw="{not json")
        with self.assertRaises(BusinessError) as ctx:
            load_spec(malformed)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_duplicate_sample_json_key_is_e_spec(self):
        raw = '{"events": [{"source_id": "a", "source_seq": 1, "value": 1, "value": 2}]}'
        path = write_tmp(None, raw=raw)
        with self.assertRaises(BusinessError) as ctx:
            load_sample_overrides(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")


if __name__ == "__main__":
    unittest.main()
