from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from columnar_diag.codec import decode_block, encode_block
from columnar_diag.errors import AppError
from columnar_diag.sample import build_sample_spec
from columnar_diag.segment import encode_spec_blocks, read_segment, write_segment
from columnar_diag.spec import parse_spec


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = build_sample_spec().to_dict()

    def _assert_spec_error(self, raw: dict, expected_field: str, expected_block: str | None = None) -> None:
        with self.assertRaises(AppError) as context:
            parse_spec(raw)
        error = context.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.field, expected_field)
        if expected_block is not None:
            self.assertEqual(error.block_id, expected_block)

    def test_duplicate_column(self) -> None:
        self.raw["columns"].append({"name": "region", "type": "int"})
        self._assert_spec_error(self.raw, "name", "<column:region>")

    def test_row_group_gap(self) -> None:
        self.raw["row_groups"][1]["start_row"] = 9
        self._assert_spec_error(self.raw, "start_row", "<row_group:rg1>")

    def test_invalid_encoding_reports_first_block(self) -> None:
        self.raw["blocks"][0]["encoding"] = "delta"
        self._assert_spec_error(self.raw, "encoding", "region@rg0")

    def test_negative_compressed_size(self) -> None:
        self.raw["blocks"][0]["compressed_bytes"] = -1
        self._assert_spec_error(self.raw, "compressed_bytes", "region@rg0")

    def test_declared_size_mismatch(self) -> None:
        self.raw["blocks"][0]["uncompressed_bytes"] += 1
        self._assert_spec_error(self.raw, "uncompressed_bytes", "region@rg0")

    def test_query_missing_column(self) -> None:
        self.raw["queries"][0]["where"] = "missing_col = 3"
        self._assert_spec_error(self.raw, "where", "query:q1")


class CodecAndSegmentTests(unittest.TestCase):
    def test_roundtrip_all_encodings_and_compression(self) -> None:
        values = (3, 3, 5, -1, 5, 5, 9, -1)
        for encoding in ("raw", "dictionary", "rle"):
            for compressed in (False, True):
                first = encode_block(
                    block_id=f"b-{encoding}-{compressed}",
                    column="value",
                    row_group="rg0",
                    encoding=encoding,
                    compressed=compressed,
                    values=values,
                )
                second = encode_block(
                    block_id=f"b-{encoding}-{compressed}",
                    column="value",
                    row_group="rg0",
                    encoding=encoding,
                    compressed=compressed,
                    values=values,
                )
                self.assertEqual(first.block_digest, second.block_digest)
                self.assertEqual(first.record_bytes, second.record_bytes)
                self.assertEqual(decode_block(first), values)
                self.assertEqual(first.pre_bytes, len(first.payload) if not compressed else first.pre_bytes)
                self.assertGreaterEqual(first.post_bytes, 0)

    def test_dictionary_order_is_numeric_ascending(self) -> None:
        block = encode_block(
            block_id="b",
            column="v",
            row_group="rg",
            encoding="dictionary",
            compressed=False,
            values=(9, -2, 5, -2, 9),
        )
        import struct

        count = struct.unpack_from("<I", block.payload, 0)[0]
        dictionary = [struct.unpack_from("<q", block.payload, 4 + index * 8)[0] for index in range(count)]
        self.assertEqual(dictionary, [-2, 5, 9])

    def test_segment_roundtrip_and_tamper_detection(self) -> None:
        spec = build_sample_spec()
        encoded = encode_spec_blocks(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment.seg"
            written = write_segment(
                path,
                spec,
                [encoded[block.id] for block in spec.blocks],
                encoding_selection_version=7,
            )
            image = read_segment(path)
            self.assertEqual(image.file_digest, written.file_digest)
            self.assertEqual(image.encoding_selection_version, 7)
            for block in spec.blocks:
                self.assertEqual(image.values(block.id), block.values)
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0x01
            path.write_bytes(data)
            with self.assertRaises(AppError) as context:
                read_segment(path)
            self.assertEqual(context.exception.code, "E_SEGMENT")


if __name__ == "__main__":
    unittest.main()
