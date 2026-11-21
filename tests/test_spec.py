import json
import unittest
from pathlib import Path

from lease_middleware.errors import E_SPEC, MiddlewareError
from lease_middleware.spec import load_spec

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class SpecValidationTest(unittest.TestCase):
    def test_valid_sample_loads(self):
        spec = load_spec(SAMPLES / "config.json")
        self.assertEqual(len(spec.configs), 3)
        self.assertEqual(len(spec.clients), 3)
        self.assertEqual(spec.configs["feature.flags"].current_version, 3)

    def _write(self, tmp: Path, payload) -> Path:
        path = tmp / "spec.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_rejects_bad_ttl_and_missing_key(self, ):
        import tempfile
        good = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
        good["configs"][0]["ttl_seconds"] = 0
        good["clients"][0]["subscribed_keys"] = ["nope"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(MiddlewareError) as ctx:
                load_spec(self._write(Path(d), good))
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertTrue(any("ttl_seconds" in e for e in ctx.exception.context["errors"]))
        self.assertTrue(any("subscribed_keys" in e for e in ctx.exception.context["errors"]))

    def test_rejects_duplicate_keys_and_bad_format(self):
        import tempfile
        good = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
        good["configs"].append(dict(good["configs"][0]))
        good["clients"][1]["format_version"] = "r9"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(MiddlewareError) as ctx:
                load_spec(self._write(Path(d), good))
        self.assertEqual(ctx.exception.code, E_SPEC)
        joined = " ".join(ctx.exception.context["errors"])
        self.assertIn("duplicate", joined)
        self.assertIn("format_version", joined)

    def test_rejects_unparseable_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(MiddlewareError) as ctx:
                load_spec(path)
        self.assertEqual(ctx.exception.code, E_SPEC)


if __name__ == "__main__":
    unittest.main()
