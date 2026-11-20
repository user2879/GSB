from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from edgecache.errors import EdgeError
from edgecache.spec import load_spec

from tests.helpers import SAMPLE_V1, load_raw, write_raw


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_valid_sample(self) -> None:
        spec = load_spec(SAMPLE_V1)
        self.assertEqual(3, len(spec.policy.nodes))
        self.assertEqual(
            ["/data/left", "/data/right", "/page/home"],
            spec.content.descendant_order("/data/root"),
        )
        self.assertEqual(
            ["/page/home"],
            spec.content.descendant_order("/api/config"),
        )

    def test_duplicate_path_is_spec_error(self) -> None:
        document = load_raw(SAMPLE_V1)
        document["content"].append(dict(document["content"][0]))
        path = write_raw(self.path / "duplicate.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path)
        self.assertEqual("E_SPEC", context.exception.code)
        self.assertEqual("/api/config", context.exception.path)
        self.assertEqual("content[].path", context.exception.field_path)

    def test_ttl_and_key_field_validation(self) -> None:
        document = load_raw(SAMPLE_V1)
        for record in document["content"]:
            if record["path"] == "/data/root":
                record["ttl_ms"] = 0
        path = write_raw(self.path / "ttl.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path)
        self.assertEqual("E_SPEC", context.exception.code)
        self.assertEqual("/data/root", context.exception.path)
        self.assertEqual("content[/data/root].ttl_ms", context.exception.field_path)

        document = load_raw(SAMPLE_V1)
        document["policy"]["key_fields"] = ["path", "path"]
        path = write_raw(self.path / "keys.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path)
        self.assertEqual("E_SPEC", context.exception.code)
        self.assertEqual("policy.key_fields[1]", context.exception.field_path)

    def test_missing_dependency_is_spec_error(self) -> None:
        document = load_raw(SAMPLE_V1)
        for record in document["content"]:
            if record["path"] == "/data/left":
                record["derived_from"] = ["/missing/value"]
        path = write_raw(self.path / "missing.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path)
        self.assertEqual("E_SPEC", context.exception.code)
        self.assertEqual("/data/left", context.exception.path)
        self.assertEqual("content[/data/left].derived_from[0]", context.exception.field_path)

    def test_cycle_code_can_be_classified_for_injection(self) -> None:
        document = load_raw(SAMPLE_V1)
        for record in document["content"]:
            if record["path"] == "/data/left":
                record["derived_from"] = ["/page/home"]
        path = write_raw(self.path / "cycle.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path, cycle_error_code="E_CYCLE")
        self.assertEqual("E_CYCLE", context.exception.code)
        self.assertEqual("/data/left", context.exception.path)
        self.assertIn("/data/left", context.exception.details["cycle"])
        self.assertEqual("E_SPEC", context.exception.details["spec_rejection"])

    def test_content_truncation_is_content_error(self) -> None:
        document = load_raw(SAMPLE_V1)
        for record in document["content"]:
            if record["path"] == "/data/root":
                raw = base64.b64decode(record["content_b64"])
                record["content_b64"] = base64.b64encode(raw[:-1]).decode("ascii")
        path = write_raw(self.path / "truncated.json", document)
        with self.assertRaises(EdgeError) as context:
            load_spec(path)
        self.assertEqual("E_CONTENT", context.exception.code)
        self.assertEqual("/data/root", context.exception.path)
        self.assertEqual("content[/data/root].content_b64", context.exception.field_path)

    def test_first_error_is_stable_after_reordering(self) -> None:
        message = None
        for reverse in (False, True):
            document = load_raw(SAMPLE_V1)
            for record in document["content"]:
                if record["path"] in {"/data/root", "/page/home"}:
                    record["ttl_ms"] = 0
            if reverse:
                document["content"].reverse()
            path = write_raw(self.path / f"stable-{reverse}.json", document)
            with self.assertRaises(EdgeError) as context:
                load_spec(path)
            if message is None:
                message = (context.exception.path, context.exception.field_path)
            else:
                self.assertEqual(message, (context.exception.path, context.exception.field_path))
        self.assertEqual(("/data/root", "content[/data/root].ttl_ms"), message)


if __name__ == "__main__":
    unittest.main()
