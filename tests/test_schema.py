import json
import tempfile
import unittest
from pathlib import Path

from colbench.errors import SchemaError
from colbench.sample_data import build_spec
from colbench.schema import load_segment_spec, parse_segment_spec


def _write(raw: dict) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "seg.json"
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    return tmp


class SchemaTests(unittest.TestCase):
    def test_valid_spec_loads(self):
        spec = build_spec("seg_a", 1)
        self.assertEqual(len(spec.row_groups), 3)
        self.assertEqual(len(spec.blocks), 12)

    def test_duplicate_column_rejected(self):
        spec = build_spec("seg_x", 1)
        raw = json.loads(Path(_write_path(spec)).read_text())
        raw["columns"][1]["name"] = raw["columns"][0]["name"]
        with self.assertRaises(SchemaError) as ctx:
            parse_segment_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")
        self.assertTrue(any(i["column"] == raw["columns"][0]["name"]
                            for i in ctx.exception.details["issues"]))

    def test_row_group_gap_rejected(self):
        spec = build_spec("seg_y", 1)
        raw = json.loads(Path(_write_path(spec)).read_text())
        raw["row_groups"][1]["start"] += 1
        with self.assertRaises(SchemaError) as ctx:
            parse_segment_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_bad_encoding_and_bytes_rejected(self):
        spec = build_spec("seg_z", 1)
        raw = json.loads(Path(_write_path(spec)).read_text())
        raw["blocks"][0]["encoding"] = "HUFFMAN"
        raw["blocks"][1]["pre_bytes"] = -3
        with self.assertRaises(SchemaError) as ctx:
            parse_segment_spec(raw)
        issues = ctx.exception.details["issues"]
        self.assertTrue(any("HUFFMAN" in i["message"] for i in issues))
        self.assertTrue(any("pre_bytes" in i["message"] for i in issues))
        self.assertIsNotNone(issues[0]["block_index"])

    def test_missing_block_rejected(self):
        spec = build_spec("seg_w", 1)
        raw = json.loads(Path(_write_path(spec)).read_text())
        raw["blocks"].pop()
        with self.assertRaises(SchemaError):
            parse_segment_spec(raw)


def _write_path(spec) -> str:
    tmp = Path(tempfile.mkdtemp()) / "seg.json"
    from colbench.schema import write_segment_spec
    write_segment_spec(spec, tmp)
    return tmp


if __name__ == "__main__":
    unittest.main()
