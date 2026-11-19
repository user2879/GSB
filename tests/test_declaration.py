from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from h2mux.declaration import parse_declaration, parse_script
from h2mux.errors import DeclarationError


ROOT = Path(__file__).resolve().parents[1]


def sample_declaration() -> dict:
    return json.loads((ROOT / "samples" / "session.json").read_text(encoding="utf-8"))


class DeclarationValidationTests(unittest.TestCase):
    def test_sample_declaration_loads(self) -> None:
        declaration = parse_declaration(sample_declaration())
        self.assertEqual(declaration.declaration_version, 7)
        self.assertEqual([item.stream_id for item in declaration.requests], [1, 3, 5, 7, 9])

    def test_negative_window_reports_exact_path(self) -> None:
        value = sample_declaration()
        value["client"]["connection_window"] = -1
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.code, "E_DECL")
        self.assertEqual(context.exception.path, "$.client.connection_window")

    def test_even_stream_id_rejected(self) -> None:
        value = sample_declaration()
        value["requests"][0]["stream_id"] = 2
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.path, "$.requests[0].stream_id")

    def test_duplicate_stream_id_rejected(self) -> None:
        value = sample_declaration()
        value["requests"][1]["stream_id"] = 1
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.path, "$.requests[1].stream_id")

    def test_missing_parent_rejected(self) -> None:
        value = sample_declaration()
        value["requests"][3]["parent_id"] = 11
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.path, "$.requests[3].parent_id")

    def test_dependency_cycle_rejected(self) -> None:
        value = sample_declaration()
        value["requests"][0]["parent_id"] = 3
        value["requests"][1]["parent_id"] = 1
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.path, "$.requests[0].parent_id")
        self.assertIn("cycle", context.exception.details)

    def test_unsupported_frame_size_rejected(self) -> None:
        value = sample_declaration()
        value["max_frame_size"] = 20000
        with self.assertRaises(DeclarationError) as context:
            parse_declaration(value)
        self.assertEqual(context.exception.path, "$.max_frame_size")

    def test_script_version_mismatch_rejected(self) -> None:
        declaration = parse_declaration(sample_declaration())
        script = json.loads((ROOT / "samples" / "frames.json").read_text(encoding="utf-8"))
        script["declaration_version"] = 8
        with self.assertRaises(DeclarationError) as context:
            parse_script(script, declaration=declaration)
        self.assertEqual(context.exception.path, "$.declaration_version")

    def test_invalid_frame_spec_path(self) -> None:
        declaration = parse_declaration(sample_declaration())
        script = {
            "script_version": 1,
            "declaration_version": 7,
            "events": [
                {
                    "event": "recv",
                    "frame": {
                        "type": "WINDOW_UPDATE",
                        "stream_id": 0,
                        "increment": 0,
                    },
                }
            ],
        }
        with self.assertRaises(DeclarationError) as context:
            parse_script(script, declaration=declaration)
        self.assertEqual(context.exception.path, "$.events[0].frame.increment")


if __name__ == "__main__":
    unittest.main()
