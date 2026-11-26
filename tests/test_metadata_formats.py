"""Requirement 2: c1/c2/c3 codecs, normalization and E_FORMAT."""
import os
import tempfile
import unittest

from shardmeta.errors import FormatError
from shardmeta.metadata import (
    C1_LINE_WIDTH,
    decode_file,
    encode_c1_line,
    encode_json_line,
    normalize_files,
)
from shardmeta.models import FORMAT_C1, FORMAT_C2, FORMAT_C3, NormalizedRecord


def record(**overrides):
    base = dict(
        object_key="obj", shard_index=1, shard_digest="a" * 16,
        merge_id="m1", shard_bytes=8, meta_version=2,
        content_check="c" * 16, parent_merge_id=None, commit_cursor=3,
        source_format=FORMAT_C3, source_line=1,
    )
    base.update(overrides)
    return NormalizedRecord(**base)


class MetadataFormatTest(unittest.TestCase):
    def test_c1_roundtrip_fixed_width(self):
        c1 = record(source_format=FORMAT_C1, merge_id=None, shard_bytes=None,
                    meta_version=None, content_check=None, parent_merge_id=None,
                    commit_cursor=None)
        line = encode_c1_line(c1)
        self.assertEqual(len(line), C1_LINE_WIDTH)
        handle = tempfile.NamedTemporaryFile("w", suffix=".c1.txt", delete=False)
        handle.write(line + "\n")
        handle.close()
        files = normalize_files([decode_file(handle.name)])
        self.assertEqual(files[0].object_key, "obj")
        self.assertIn("merge_id", files[0].filled_fields)
        self.assertEqual(files[0].fill_basis["merge_id"], "synthetic:line+digest")

    def test_bad_c1_width_reports_line(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".c1.txt", delete=False)
        handle.write("too-short\n")
        handle.close()
        with self.assertRaises(FormatError) as ctx:
            normalize_files([decode_file(handle.name)])
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertEqual(ctx.exception.line, 1)

    def test_c2_and_c3_roundtrip(self):
        c2 = record(source_format=FORMAT_C2, content_check=None,
                    parent_merge_id=None, commit_cursor=None)
        line = encode_json_line(c2, FORMAT_C2)
        handle = tempfile.NamedTemporaryFile("w", suffix=".c2.jsonl", delete=False)
        handle.write(line + "\n")
        handle.close()
        normalized = normalize_files([decode_file(handle.name)])[0]
        self.assertIn("content_check", normalized.filled_fields)
        self.assertNotIn("merge_id", normalized.filled_fields)

    def test_c3_real_values_not_overwritten(self):
        c3 = record()
        line = encode_json_line(c3, FORMAT_C3)
        handle = tempfile.NamedTemporaryFile("w", suffix=".c3.jsonl", delete=False)
        handle.write(line + "\n")
        handle.close()
        normalized = normalize_files([decode_file(handle.name)])[0]
        self.assertEqual(normalized.filled_fields, frozenset())
        self.assertEqual(normalized.content_check, "c" * 16)
        self.assertEqual(normalized.commit_cursor, 3)

    def test_duplicate_merge_id(self):
        c3a = record(merge_id="dup", object_key="a", shard_index=0)
        c3b = record(merge_id="dup", object_key="b", shard_index=0,
                     shard_digest="b" * 16)
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "x.c3.jsonl")
        with open(path, "w") as handle:
            handle.write(encode_json_line(c3a, FORMAT_C3) + "\n")
            handle.write(encode_json_line(c3b, FORMAT_C3) + "\n")
        with self.assertRaises(FormatError) as ctx:
            normalize_files([decode_file(path)])
        self.assertEqual(ctx.exception.field, "merge_id")

    def test_missing_parent_reference(self):
        child = record(merge_id="child", parent_merge_id="ghost")
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "x.c3.jsonl")
        with open(path, "w") as handle:
            handle.write(encode_json_line(child, FORMAT_C3) + "\n")
        with self.assertRaises(FormatError) as ctx:
            normalize_files([decode_file(path)])
        self.assertEqual(ctx.exception.field, "parent_merge_id")

    def test_parent_cycle_detected(self):
        a = record(merge_id="a", parent_merge_id="b", object_key="o", shard_index=0)
        b = record(merge_id="b", parent_merge_id="a", object_key="o", shard_index=0,
                   shard_digest="b" * 16)
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "x.c3.jsonl")
        with open(path, "w") as handle:
            handle.write(encode_json_line(a, FORMAT_C3) + "\n")
            handle.write(encode_json_line(b, FORMAT_C3) + "\n")
        with self.assertRaises(FormatError) as ctx:
            normalize_files([decode_file(path)])
        self.assertEqual(ctx.exception.field, "parent_merge_id")


if __name__ == "__main__":
    unittest.main()
