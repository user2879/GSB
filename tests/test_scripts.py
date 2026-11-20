from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from isolation_lab.errors import BusinessError
from isolation_lab.models import (
    CURRENT_FORMAT_VERSION,
    DEFAULT_CONCURRENCY,
    DEFAULT_ISOLATION_LEVEL,
    DEFAULT_LOCK_WAIT_BUDGET_MS,
    DEFAULT_OPERATION_TIMEOUT_MS,
)
from isolation_lab.scripts import load_script, normalize_document


ROOT = Path(__file__).resolve().parent.parent


def minimal_v1() -> dict:
    return {
        "format_version": 1,
        "script_id": "minimal-v1",
        "initial_rows": {"row:a": 0},
        "sessions": [
            {
                "session_id": "S1",
                "operations": [
                    {"op_no": 1, "type": "read", "row_key": "row:a"},
                    {"op_no": 2, "type": "write", "row_key": "row:a", "value": 1},
                ],
            }
        ],
    }


class ScriptVersionTests(unittest.TestCase):
    def test_samples_load_across_three_versions(self) -> None:
        expected = {
            "current-v3.json": 3,
            "legacy-v1.json": 1,
            "legacy-v2.json": 2,
            "lock-timeout-v3.json": 3,
            "deadlock-v3.json": 3,
        }
        for filename, source_version in expected.items():
            script = load_script(ROOT / "samples" / filename)
            self.assertEqual(script.source_version, source_version)
            self.assertEqual(script.normalized_version, CURRENT_FORMAT_VERSION)
            self.assertGreater(len(script.sessions), 0)

    def test_v1_defaults_and_source_marker(self) -> None:
        script = normalize_document(minimal_v1())
        session = script.sessions[0]
        self.assertEqual(script.source_version, 1)
        self.assertEqual(script.normalized_version, 3)
        self.assertEqual(session.isolation_level, DEFAULT_ISOLATION_LEVEL)
        self.assertEqual(session.lock_wait_budget_ms, DEFAULT_LOCK_WAIT_BUDGET_MS)
        self.assertEqual(session.concurrency, DEFAULT_CONCURRENCY)
        self.assertEqual(session.operation_timeout_ms, DEFAULT_OPERATION_TIMEOUT_MS)
        self.assertEqual(session.lock_order, ("row:a",))
        self.assertIn("format_version:1->3", script.upgrade_changes)

    def test_upgraded_document_can_be_loaded_again(self) -> None:
        first = normalize_document(minimal_v1())
        upgraded = first.to_dict(include_source=True)
        second = normalize_document(upgraded)
        self.assertEqual(second.source_version, 1)
        self.assertEqual(second.normalized_version, 3)
        self.assertEqual(second.semantic_digest, first.semantic_digest)
        self.assertEqual(second.upgrade_changes, first.upgrade_changes)

    def test_unknown_version_rejected_with_path(self) -> None:
        document = minimal_v1()
        document["format_version"] = 99
        with self.assertRaises(BusinessError) as captured:
            normalize_document(document)
        error = captured.exception
        self.assertEqual(error.code, "E_SCRIPT")
        self.assertEqual(error.path, "$.format_version")
        self.assertEqual(error.details["format_version"], 99)

    def test_missing_session_field_reports_session_and_path(self) -> None:
        document = {
            "format_version": 2,
            "script_id": "bad-v2",
            "initial_rows": {"row:a": 0},
            "sessions": [
                {
                    "session_id": "BROKEN",
                    "lock_wait_budget_ms": 10,
                    "operations": [
                        {"op_no": 1, "type": "read", "row_key": "row:a"}
                    ],
                }
            ],
        }
        with self.assertRaises(BusinessError) as captured:
            normalize_document(document)
        error = captured.exception
        self.assertEqual(error.code, "E_SCRIPT")
        self.assertEqual(error.session_id, "BROKEN")
        self.assertEqual(error.path, "$.sessions[0].isolation_level")

    def test_wrong_field_type_is_rejected(self) -> None:
        document = minimal_v1()
        document["sessions"][0]["operations"][1]["op_no"] = "2"
        with self.assertRaises(BusinessError) as captured:
            normalize_document(document)
        error = captured.exception
        self.assertEqual(error.code, "E_SCRIPT")
        self.assertEqual(error.session_id, "S1")
        self.assertIn("expected an integer", error.message)
        self.assertIn("operations[1].op_no", error.path)

    def test_version_specific_unknown_field_is_rejected(self) -> None:
        document = minimal_v1()
        document["sessions"][0]["isolation_level"] = "read_committed"
        with self.assertRaises(BusinessError) as captured:
            normalize_document(document)
        error = captured.exception
        self.assertEqual(error.code, "E_SCRIPT")
        self.assertEqual(error.session_id, "S1")
        self.assertIn("unknown field: isolation_level", error.message)

    def test_json_parse_error_is_script_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(BusinessError) as captured:
                load_script(path)
        self.assertEqual(captured.exception.code, "E_SCRIPT")
        self.assertEqual(captured.exception.path, "$")


if __name__ == "__main__":
    unittest.main()
