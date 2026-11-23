from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from imv.errors import BusinessError
from imv.model import load_base_table, load_dataset


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.base = self.dir / "base.json"
        self.views = self.dir / "views.json"
        self.changes = self.dir / "changes.ndjson"
        shutil.copy(SAMPLES / "base_table.json", self.base)
        shutil.copy(SAMPLES / "views.json", self.views)
        shutil.copy(SAMPLES / "changes.ndjson", self.changes)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _mutate_json(self, path: Path, mutator) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutator(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _assert_schema(self, callable_obj) -> BusinessError:
        with self.assertRaises(BusinessError) as caught:
            callable_obj()
        self.assertEqual(caught.exception.code, "E_SCHEMA")
        self.assertTrue(caught.exception.location)
        return caught.exception

    def test_duplicate_column_is_rejected_at_first_location(self) -> None:
        self._mutate_json(self.base, lambda payload: payload["columns"][1].__setitem__("name", "sale_id"))
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertIn("columns[1].name", error.location["path"])

    def test_unknown_group_and_aggregate_are_rejected(self) -> None:
        self._mutate_json(self.views, lambda payload: payload["views"][0]["group_by"].__setitem__(0, "missing_column"))
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertEqual(error.location["view_id"], "v_region")
        self._mutate_json(self.views, lambda payload: payload["views"][0].__setitem__("group_by", ["region_id"]))
        self._mutate_json(self.views, lambda payload: payload["views"][0]["aggregates"][0].__setitem__("func", "average"))
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertIn("aggregates[0].func", error.location["path"])

    def test_change_sequence_duplicate_and_gap_are_rejected(self) -> None:
        lines = self.changes.read_text(encoding="utf-8").splitlines()
        second = json.loads(lines[1])
        second["seq"] = 1
        lines[1] = json.dumps(second, ensure_ascii=False)
        self.changes.write_text("\n".join(lines) + "\n", encoding="utf-8")
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertEqual(error.location["line"], 2)
        self.assertEqual(error.location["path"], "$.seq")

        lines[1] = json.dumps({**second, "seq": 4}, ensure_ascii=False)
        self.changes.write_text("\n".join(lines) + "\n", encoding="utf-8")
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertEqual(error.location["line"], 2)

    def test_update_and_delete_require_existing_row_id(self) -> None:
        lines = self.changes.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["row_id"] = 999
        first["row"]["sale_id"] = 999
        lines[0] = json.dumps(first, ensure_ascii=False)
        self.changes.write_text("\n".join(lines) + "\n", encoding="utf-8")
        error = self._assert_schema(lambda: load_dataset(self.base, self.views, self.changes))
        self.assertEqual(error.location["row_id"], 999)
        self.assertEqual(error.location["line"], 1)

    def test_null_bitmap_must_be_consistent(self) -> None:
        self._mutate_json(self.base, lambda payload: payload["null_bitmap"].__setitem__(1, {"row_id": 5, "columns": []}))
        error = self._assert_schema(lambda: load_base_table(self.base))
        self.assertEqual(error.code, "E_SCHEMA")


if __name__ == "__main__":
    unittest.main()
