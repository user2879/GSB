from __future__ import annotations

import json
import unittest
from pathlib import Path

from config_ledger.errors import SpecValidationError
from config_ledger.service import DiagnosticService
from config_ledger.spec import load_spec, parse_spec_data

from .helpers import log_records, make_spec_data, temp_workspace, write_json


ROOT = Path(__file__).resolve().parent.parent


class SpecValidationTests(unittest.TestCase):
    def test_non_numeric_string_expected_is_valid(self) -> None:
        spec = load_spec(ROOT / "samples" / "string_expected" / "spec.json")
        self.assertEqual(spec.keys["deployment.color"].expected, "blue")
        self.assertIsInstance(spec.keys["deployment.color"].expected, str)

    def test_invalid_sample_reports_both_field_paths_and_node_scope(self) -> None:
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(ROOT / "samples" / "invalid_expected" / "spec.json")
        exc = caught.exception
        self.assertEqual(exc.code, "E_SPEC")
        paths = {issue.field_path for issue in exc.issues}
        self.assertIn('keys["array.expected"].expected', paths)
        self.assertIn('keys["boolean.expected"].expected', paths)
        self.assertTrue(all(issue.node_id == "<keys>" for issue in exc.issues))

    def test_node_and_global_issues_are_collected_without_partial_spec(self) -> None:
        data = make_spec_data(
            {"node-a": 45001, "node-b": 45001},
            {"k": {"expected": True, "quorum_bp": 10001}},
        )
        data["nodes"][1]["host"] = "0.0.0.0"
        with self.assertRaises(SpecValidationError) as caught:
            parse_spec_data(data)
        paths = {issue.field_path for issue in caught.exception.issues}
        node_ids = {issue.node_id for issue in caught.exception.issues}
        self.assertIn("nodes[1].port", paths)
        self.assertIn("nodes[1].host", paths)
        self.assertIn('keys["k"].expected', paths)
        self.assertIn('keys["k"].quorum_bp', paths)
        self.assertIn("node-b", node_ids)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with temp_workspace() as temp:
            path = temp / "duplicate.json"
            path.write_text('{"name":"a","name":"b"}', encoding="utf-8")
            with self.assertRaises(SpecValidationError) as caught:
                load_spec(path)
            self.assertIn("duplicate JSON object key", str(caught.exception))

    def test_failed_reload_does_not_replace_in_memory_or_ledger_state(self) -> None:
        with temp_workspace() as temp:
            valid = make_spec_data(
                {"node-a": 45011}, {"key": {"expected": "ok", "quorum_bp": 10000}}
            )
            valid_path = write_json(temp / "valid.json", valid)
            invalid_path = write_json(temp / "invalid.json", {"nodes": "broken"})
            log_path = temp / "logs.jsonl"
            service = DiagnosticService(
                valid_path, ledger_path=temp / "ledger.sqlite", log_path=log_path
            )
            try:
                before_spec = service.spec
                before_meta = service.ledger.current_meta()
                with self.assertRaises(SpecValidationError):
                    service.reload_spec(invalid_path)
                self.assertEqual(service.spec, before_spec)
                self.assertEqual(service.ledger.current_meta(), before_meta)
                error_events = [r for r in log_records(log_path) if r["code"] == "E_SPEC"]
                self.assertTrue(error_events)
                self.assertIn("issues", error_events[-1])
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
