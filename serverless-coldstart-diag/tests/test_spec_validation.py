import json
import unittest
from pathlib import Path

from coldstart_diag.errors import SpecError
from coldstart_diag.samples import base_manifest
from coldstart_diag.spec import load_manifest, validate_manifest


class SpecValidationTest(unittest.TestCase):
    def test_valid_manifest(self):
        doc = base_manifest(1, 1, 10000, 2500)
        self.assertEqual(validate_manifest(json.loads(json.dumps(doc)))["declaration_version"], 1)

    def _expect(self, mutator):
        doc = base_manifest(1, 1, 10000, 2500)
        mutator(doc)
        with self.assertRaises(SpecError) as ctx:
            validate_manifest(doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        return ctx.exception.details

    def test_memory_out_of_range(self):
        details = self._expect(lambda d: d["functions"][0].__setitem__("memory_mib", 3))
        self.assertTrue(any("functions[0].memory_mib" == e["path"] for e in details))

    def test_non_integer(self):
        details = self._expect(lambda d: d["functions"][0].__setitem__("init_ms", 1.5))
        self.assertTrue(any("init_ms" in e["path"] for e in details))

    def test_duplicate_function_version(self):
        doc = base_manifest(1, 1, 10000, 2500)
        doc["functions"].append(json.loads(json.dumps(doc["functions"][0])))
        with self.assertRaises(SpecError) as ctx:
            validate_manifest(doc)
        self.assertTrue(any("唯一" in e["reason"] for e in ctx.exception.details))

    def test_non_monotonic_version(self):
        doc = base_manifest(1, 2, 10000, 2500)
        clone = json.loads(json.dumps(doc["functions"][0]))
        clone["function_version"] = 1
        doc["functions"].append(clone)
        with self.assertRaises(SpecError) as ctx:
            validate_manifest(doc)
        self.assertTrue(any("单调递增" in e["reason"] for e in ctx.exception.details))

    def test_minute_gap_and_permyriad(self):
        details = self._expect(lambda d: d["call_sequence"][2].__setitem__("minute", 9))
        self.assertTrue(any("连续" in e["reason"] for e in details))
        details = self._expect(lambda d: d["call_sequence"][0].__setitem__("burst_permyriad", 20000))
        self.assertTrue(any("burst_permyriad" in e["path"] for e in details))

    def test_negative_calls(self):
        details = self._expect(lambda d: d["call_sequence"][0].__setitem__("calls", -1))
        self.assertTrue(any("calls" in e["path"] for e in details))

    def test_scale_must_be_power_of_ten(self):
        details = self._expect(lambda d: d.__setitem__("scale", 500))
        self.assertTrue(any(e["path"] == "scale" for e in details))

    def test_dangling_reference(self):
        def mutate(d):
            d["call_sequence"][0]["function_version"] = 99
        details = self._expect(mutate)
        self.assertTrue(any("未声明" in e["reason"] for e in details))

    def test_missing_file(self):
        with self.assertRaises(SpecError):
            load_manifest(Path("/nonexistent/manifest.json"))


if __name__ == "__main__":
    unittest.main()
