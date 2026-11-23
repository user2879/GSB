"""E_SPEC strong-validation tests."""

import json
import os
import tempfile
import unittest

from tests.helpers import make_spec_files
from workbench import E_SPEC
from workbench.errors import DiagError
from workbench.spec import load_spec


def _rewrite(tmpdir, name, mutate):
    path = os.path.join(tmpdir, name)
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    mutate(obj)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wbspec-")
        make_spec_files(self.tmp)

    def _expect_espec(self, name, mutate, field):
        _rewrite(self.tmp, name, mutate)
        with self.assertRaises(DiagError) as ctx:
            load_spec(self.tmp)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertEqual(ctx.exception.field, field)
        self.assertIsNotNone(ctx.exception.line)
        self.assertGreaterEqual(ctx.exception.line, 1)

    def test_valid_sample_loads(self):
        spec = load_spec(self.tmp)
        self.assertEqual(len(spec.services), 2)
        self.assertEqual(len(spec.requests), 8)
        self.assertEqual(spec.current_version("svc-a"), 2)

    def test_duplicate_service_id(self):
        def mutate(obj):
            obj["services"].append(dict(obj["services"][0]))
        self._expect_espec("services.json", mutate, "service_id")

    def test_non_loopback_host(self):
        def mutate(obj):
            obj["services"][0]["host"] = "localhost"
        self._expect_espec("services.json", mutate, "host")

    def test_duplicate_port(self):
        def mutate(obj):
            obj["services"][1]["port"] = obj["services"][0]["port"]
        self._expect_espec("services.json", mutate, "port")

    def test_non_positive_capacity(self):
        def mutate(obj):
            obj["services"][0]["capacity_tokens"] = 0
        self._expect_espec("services.json", mutate, "capacity_tokens")

    def test_non_integer_refill_interval(self):
        def mutate(obj):
            obj["services"][0]["refill_interval_ms"] = 12.5
        self._expect_espec("services.json", mutate, "refill_interval_ms")

    def test_non_positive_window(self):
        def mutate(obj):
            obj["services"][0]["signature_window_ms"] = -1
        self._expect_espec("services.json", mutate, "signature_window_ms")

    def test_key_versions_not_increasing(self):
        def mutate(obj):
            obj["keys"]["svc-a"][1]["version"] = 1
        self._expect_espec("keys.json", mutate, "version")

    def test_retire_null_on_non_current(self):
        def mutate(obj):
            obj["keys"]["svc-a"][0]["retire_ms"] = None
        self._expect_espec("keys.json", mutate, "retire_ms")

    def test_request_seq_not_contiguous(self):
        def mutate(obj):
            obj["requests"][3]["seq"] = 99
        self._expect_espec("requests.json", mutate, "seq")

    def test_arrival_not_monotonic(self):
        def mutate(obj):
            obj["requests"][5]["arrival_ms"] = 5
        self._expect_espec("requests.json", mutate, "arrival_ms")

    def test_unknown_service_reference(self):
        def mutate(obj):
            obj["requests"][0]["service_id"] = "svc-zzz"
        self._expect_espec("requests.json", mutate, "service_id")

    def test_unknown_bucket_key(self):
        def mutate(obj):
            obj["requests"][0]["bucket_key"] = "nope"
        self._expect_espec("requests.json", mutate, "bucket_key")

    def test_first_illegal_field_reported(self):
        def mutate(obj):
            obj["services"][0]["capacity_tokens"] = 0
            obj["services"][1]["port"] = 80
        _rewrite(self.tmp, "services.json", mutate)
        with self.assertRaises(DiagError) as ctx:
            load_spec(self.tmp)
        self.assertEqual(ctx.exception.field, "capacity_tokens")


if __name__ == "__main__":
    unittest.main()
