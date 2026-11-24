from __future__ import annotations

import json
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from colmv.engine import MaterializationEngine
from colmv.errors import ColmvError
from colmv.schema import load_config

from tests.helpers import copy_samples, load_json, sample_paths, write_json


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="colmv-validation-")
        self.samples = copy_samples(Path(self.temp.name))
        self.paths = sample_paths(self.samples)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_view_error(self, view_id: str, field: str) -> None:
        with self.assertRaises(ColmvError) as context:
            load_config(self.paths["catalog"], self.paths["views"])
        self.assertEqual(context.exception.code, "E_VIEW")
        self.assertEqual(context.exception.view_id, view_id)
        self.assertEqual(context.exception.field, field)

    def test_duplicate_view_id_is_e_view(self) -> None:
        views = load_json(Path(self.paths["views"]))
        views["views"][1]["view_id"] = views["views"][0]["view_id"]
        write_json(Path(self.paths["views"]), views)
        self.assert_view_error("sales_by_region", "view_id")

    def test_unknown_group_column_is_e_view(self) -> None:
        views = load_json(Path(self.paths["views"]))
        views["views"][0]["group_by"] = ["missing_column"]
        write_json(Path(self.paths["views"]), views)
        self.assert_view_error("sales_by_region", "group_by[0]")

    def test_unsupported_aggregate_is_e_view(self) -> None:
        views = load_json(Path(self.paths["views"]))
        views["views"][0]["aggregates"] = ["count", "median"]
        write_json(Path(self.paths["views"]), views)
        self.assert_view_error("sales_by_region", "aggregates[1]")

    def test_page_directory_gap_is_e_view_with_first_view(self) -> None:
        catalog = load_json(Path(self.paths["catalog"]))
        catalog["pages"][1]["start_row"] = 9
        catalog["pages"][1]["row_count"] = catalog["pages"][1]["end_row"] - 9
        write_json(Path(self.paths["catalog"]), catalog)
        self.assert_view_error("sales_by_region", "base.pages[1].start_row")

    def test_base_column_error_uses_first_declared_view_id(self) -> None:
        catalog = load_json(Path(self.paths["catalog"]))
        catalog["table"]["columns"][0]["type"] = "text"
        write_json(Path(self.paths["catalog"]), catalog)
        self.assert_view_error("sales_by_region", "base.columns[0].type")

    def test_page_null_bitmap_length_is_e_view(self) -> None:
        page_path = self.samples / "base" / "pages" / "p000.col"
        content = page_path.read_bytes()
        magic = b"COLMVPG1\n"
        header_length = struct.unpack("<I", content[len(magic) : len(magic) + 4])[0]
        header_start = len(magic) + 4
        header_end = header_start + header_length
        header = json.loads(content[header_start:header_end].decode("utf-8"))
        payload = content[header_end:]
        header["columns"][0]["null_bitmap_length"] = 7
        new_header = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        page_path.write_bytes(magic + struct.pack("<I", len(new_header)) + new_header + payload)
        with self.assertRaises(ColmvError) as context:
            load_config(self.paths["catalog"], self.paths["views"])
        self.assertEqual(context.exception.code, "E_VIEW")
        self.assertEqual(context.exception.view_id, "sales_by_region")
        self.assertIn("null_bitmap", context.exception.field or "")

    def test_non_finite_float_is_e_view(self) -> None:
        page_path = self.samples / "base" / "pages" / "p000.col"
        content = bytearray(page_path.read_bytes())
        magic = b"COLMVPG1\n"
        header_length = struct.unpack("<I", content[len(magic) : len(magic) + 4])[0]
        header_start = len(magic) + 4
        header_end = header_start + header_length
        header = json.loads(content[header_start:header_end].decode("utf-8"))
        amount = next(column for column in header["columns"] if column["name"] == "amount")
        absolute = header_end + amount["data_offset"]
        content[absolute : absolute + 8] = struct.pack("<d", float("nan"))
        payload = bytes(content[header_end:])
        amount["data_sha256"] = hashlib.sha256(
            payload[amount["data_offset"] : amount["data_offset"] + amount["data_length"]]
        ).hexdigest()
        new_header = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        page_path.write_bytes(magic + struct.pack("<I", len(new_header)) + new_header + payload)
        with self.assertRaises(ColmvError) as context:
            load_config(self.paths["catalog"], self.paths["views"])
        self.assertEqual(context.exception.code, "E_VIEW")
        self.assertEqual(context.exception.view_id, "sales_by_region")
        self.assertIn("amount", context.exception.field or "")

    def test_truncated_page_is_e_integrity(self) -> None:
        page_path = self.samples / "base" / "pages" / "p000.col"
        content = page_path.read_bytes()
        page_path.write_bytes(content[: len(content) // 2])
        with self.assertRaises(ColmvError) as context:
            load_config(self.paths["catalog"], self.paths["views"])
        self.assertEqual(context.exception.code, "E_INTEGRITY")

    def test_zero_budget_is_e_budget(self) -> None:
        config = load_config(self.paths["catalog"], self.paths["views"])
        with self.assertRaises(ColmvError) as context:
            MaterializationEngine(config, budget=0)
        self.assertEqual(context.exception.code, "E_BUDGET")


if __name__ == "__main__":
    unittest.main()
