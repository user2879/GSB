from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from planner_app.errors import AppError
from planner_app.spec import load_workspace


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(WORKSPACE.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, data) -> Path:
        path = Path(self.temp.name) / "workspace.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def assert_spec_error(self, data, path_fragment: str) -> None:
        with self.assertRaises(AppError) as raised:
            load_workspace(self.write(data))
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertIn(path_fragment, raised.exception.json_path or "")

    def test_valid_sample_loads_atomically(self) -> None:
        workspace = load_workspace(WORKSPACE)
        self.assertEqual([table.name for table in workspace.tables], ["customers", "orders"])
        self.assertEqual(len(workspace.queries), 2)
        self.assertEqual(len(workspace.edit_batches), 5)

    def test_duplicate_table_name_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["tables"][1]["name"] = "customers"
        self.assert_spec_error(raw, "tables[1].name")

    def test_duplicate_column_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["tables"][0]["columns"][1]["name"] = "id"
        self.assert_spec_error(raw, "tables[0].columns[1].name")

    def test_null_bitmap_length_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["tables"][0]["columns"][2]["nulls"].pop()
        self.assert_spec_error(raw, "tables[0].columns[2].nulls")

    def test_unknown_query_column_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["queries"][0]["sql"] = raw["queries"][0]["sql"].replace("status_code", "no_such_column", 1)
        self.assert_spec_error(raw, "queries[0].sql")

    def test_missing_edit_target_rejected_as_whole_spec(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["edit_batches"][0]["edits"][0]["target"] = "missing_node"
        self.assert_spec_error(raw, "edit_batches[0].edits[0]")

    def test_unsupported_edit_type_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["edit_batches"][0]["edits"][0]["type"] = "drop_everything"
        self.assert_spec_error(raw, "edit_batches[0].edits[0]")

    def test_non_numeric_value_rejected_at_first_row(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["tables"][1]["columns"][0]["values"][2] = "not-an-int"
        self.assert_spec_error(raw, "tables[1].columns[0].values[2]")


if __name__ == "__main__":
    unittest.main()
