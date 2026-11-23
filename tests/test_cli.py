from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from config_pipeline.cli import main
from tests.helpers import manifest


class CliTests(unittest.TestCase):
    def invoke(self, argv: list[str]) -> tuple[int, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(argv)
        return code, json.loads(output.getvalue())

    def test_validate_and_manifest_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text(json.dumps(manifest(ports=(41401,))), encoding="utf-8")
            code, payload = self.invoke(["validate", str(valid)])
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["shards"], 3)

            invalid_data = manifest(ports=(41402,))
            invalid_data["clusters"][0]["host"] = "0.0.0.0"
            invalid = root / "invalid.json"
            invalid.write_text(json.dumps(invalid_data), encoding="utf-8")
            code, payload = self.invoke(["validate", str(invalid)])
            self.assertEqual(code, 2)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["code"], "E_MANIFEST")
            self.assertEqual(payload["field"], "clusters[0].host")

    def test_compile_and_divergence_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_data = manifest(ports=(41411,))
            right_data = json.loads(json.dumps(left_data))
            right_data["revisions"][0]["values"]["base"] = 11
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(json.dumps(left_data), encoding="utf-8")
            right.write_text(json.dumps(right_data), encoding="utf-8")

            code, plan = self.invoke(["compile", str(left)])
            self.assertEqual(code, 0)
            self.assertIn("plan_digest", plan)
            self.assertEqual(len(plan["shards"]), 3)

            code, payload = self.invoke(["compare", str(left), str(right)])
            self.assertEqual(code, 6)
            self.assertEqual(payload["code"], "E_DIVERGE")
            self.assertIsNotNone(payload.get("shard_id"))


if __name__ == "__main__":
    unittest.main()
