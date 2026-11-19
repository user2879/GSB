from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vecflow.errors import AppError
from vecflow.expression import normalize_expression, parse_expression
from vecflow.model import parse_data_payload, parse_tasks_payload, workspace_state_from_parts
from tests.helpers import data_payload, make_workspace, tasks_payload


class ExpressionTests(unittest.TestCase):
    def test_whitespace_and_redundant_parentheses_normalize_identically(self) -> None:
        left = parse_expression(" ((x + y) * (flag ? 2 : 3)) ")
        right = parse_expression("(x+y)*((flag?2:3))")
        self.assertEqual(left.normalized, right.normalized)
        self.assertEqual(left.structural_digest, right.structural_digest)

    def test_if_function_is_canonical_ternary(self) -> None:
        function_form = parse_expression("if(flag, x / (d + 1.0), 0.0)")
        operator_form = parse_expression("flag ? x / (d + 1.0) : 0.0")
        self.assertEqual(function_form.normalized, operator_form.normalized)
        self.assertEqual(function_form.structural_digest, operator_form.structural_digest)
        self.assertEqual(function_form.normalized, "flag ? x / (d + 1.0) : 0.0")

    def test_nonredundant_parentheses_are_preserved(self) -> None:
        self.assertEqual(normalize_expression("a - (b - c)"), "a - (b - c)")
        self.assertEqual(normalize_expression("(a - b) - c"), "a - b - c")

    def test_logical_aliases(self) -> None:
        self.assertEqual(normalize_expression("!(a > 1) && b"), "not (a > 1) and b")
        self.assertEqual(normalize_expression("not a == b"), "not (a == b)")
        self.assertEqual(normalize_expression("not (a == b)"), "not (a == b)")


class ModelValidationTests(unittest.TestCase):
    def test_duplicate_columns_are_rejected(self) -> None:
        payload = data_payload()
        payload["blocks"][0]["columns"].append(dict(payload["blocks"][0]["columns"][0]))
        with self.assertRaises(AppError) as raised:
            parse_data_payload(payload)
        self.assertEqual(raised.exception.code, "E_DATA")

    def test_null_bitmap_length_is_e_nullmap(self) -> None:
        payload = data_payload()
        payload["blocks"][0]["columns"][0]["null_bitmap"] = [0, 1]
        with self.assertRaises(AppError) as raised:
            parse_data_payload(payload)
        self.assertEqual(raised.exception.code, "E_NULLMAP")

    def test_unmarked_null_value_is_e_data(self) -> None:
        payload = data_payload()
        payload["blocks"][0]["columns"][0]["values"][0] = None
        with self.assertRaises(AppError) as raised:
            parse_data_payload(payload)
        self.assertEqual(raised.exception.code, "E_DATA")

    def test_unknown_expression_column_is_e_data(self) -> None:
        blocks = parse_data_payload(data_payload())
        tasks = parse_tasks_payload(tasks_payload(expression="missing + 1"))
        with self.assertRaises(AppError) as raised:
            workspace_state_from_parts(blocks, tasks)
        self.assertEqual(raised.exception.code, "E_DATA")

    def test_duplicate_task_ids_are_rejected(self) -> None:
        payload = tasks_payload()
        payload["tasks"].append(dict(payload["tasks"][0]))
        with self.assertRaises(AppError) as raised:
            parse_tasks_payload(payload)
        self.assertEqual(raised.exception.code, "E_DATA")

    def test_invalid_editor_save_is_e_data_and_does_not_replace_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = make_workspace(Path(td))
            version = workspace.parameter_version
            digest = workspace.content_digest
            invalid = data_payload()
            invalid["blocks"][0]["columns"][1]["type"] = "decimal"
            with self.assertRaises(AppError) as raised:
                workspace.replace_data_json(json.dumps(invalid), persist=False)
            self.assertEqual(raised.exception.code, "E_DATA")
            self.assertEqual(workspace.parameter_version, version)
            self.assertEqual(workspace.content_digest, digest)
            self.assertIn('"name": "x"', workspace.data_json())

    def test_editor_nullmap_failure_is_wrapped_as_e_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = make_workspace(Path(td))
            version = workspace.parameter_version
            invalid = data_payload()
            invalid["blocks"][0]["columns"][0]["null_bitmap"] = [0, 1]
            with self.assertRaises(AppError) as raised:
                workspace.replace_data_json(json.dumps(invalid), persist=False)
            self.assertEqual(raised.exception.code, "E_DATA")
            self.assertEqual(raised.exception.details["cause_code"], "E_NULLMAP")
            self.assertEqual(workspace.parameter_version, version)

    def test_semantically_identical_expression_edit_does_not_increment_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = make_workspace(Path(td))
            version = workspace.parameter_version
            payload = workspace.tasks_payload()
            payload["tasks"][0]["expression"] = "  flag?(x/d):0.0"
            result = workspace.replace_tasks_json(json.dumps(payload), persist=False)
            self.assertFalse(result.changed)
            self.assertEqual(workspace.parameter_version, version)

    def test_real_edit_increments_version_and_marks_current_run_stale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = make_workspace(Path(td))

            class DummyRun:
                stale = False

                def mark_stale(self) -> None:
                    self.stale = True

            workspace.current_run = DummyRun()
            old_version = workspace.parameter_version
            result = workspace.update_global_budget(2, persist=False)
            self.assertTrue(result.changed)
            self.assertEqual(workspace.parameter_version, old_version + 1)
            self.assertTrue(workspace.current_run.stale)


if __name__ == "__main__":
    unittest.main()
