"""E_FORMAT validation and b1/b2/b3 codec tests."""
import unittest

from colval.errors import FormatError
from colval.formats import CURRENT_FORMAT, decode_block, encode_block
from colval.formats import make_bitmap, upgrade_block
from colval.store import block_from_columns


def sample_block():
    return block_from_columns("blk-t", [
        ("a", "int64", make_bitmap([], 4), [1, 2, 3, 4]),
        ("b", "float64", make_bitmap([1], 4), [0.5, 0.0, 2.5, -1.25]),
    ], quant_scale_exp=3, batch_max_rows=1024)


class TestFormats(unittest.TestCase):
    def test_roundtrip_all_versions(self):
        blk = sample_block()
        for version in ("b1", "b2", "b3"):
            if version == "b1":
                blk1 = block_from_columns("blk-t", [
                    ("a", "int64", make_bitmap([], 4), [1, 2, 3, 4]),
                    ("b", "float64", make_bitmap([], 4), [0.5, 1.5, 2.5, -1.25]),
                ], 3, 1024)
                dec = decode_block(encode_block(blk1, "b1"))
                self.assertEqual(dec.format_version, "b1")
                self.assertEqual(dec.columns[1].dtype, "float64")
                continue
            dec = decode_block(encode_block(blk, version))
            self.assertEqual(dec.row_count, 4)
            self.assertEqual(dec.columns[1].null_bitmap, make_bitmap([1], 4))

    def test_upgrade_b1_to_b3(self):
        blk1 = block_from_columns("blk-t", [
            ("a", "int64", make_bitmap([], 3), [7, 8, 9]),
        ], 3, 1024)
        old = decode_block(encode_block(blk1, "b1"))
        up = upgrade_block(old)
        self.assertEqual(up.format_version, CURRENT_FORMAT)
        self.assertEqual(up.quant_scale_exp, 3)
        self.assertEqual(up.batch_max_rows, 1024)
        self.assertEqual(up.columns[0].dtype, "int64")
        self.assertEqual(up.columns[0].null_bitmap, make_bitmap([], 3))

    def test_unknown_version_rejected(self):
        with self.assertRaises(FormatError) as cm:
            decode_block('{"format":"b9","block_id":"blk-x","row_count":1,'
                         '"columns":[{"name":"a","values":[1]}]}')
        self.assertEqual(cm.exception.code, "E_FORMAT")
        self.assertEqual(cm.exception.context["block_id"], "blk-x")
        self.assertEqual(cm.exception.context["field"], "format")

    def test_missing_field_rejected(self):
        with self.assertRaises(FormatError) as cm:
            decode_block('{"format":"b2","block_id":"blk-y","row_count":1,'
                         '"columns":[{"name":"a","values":[1]}]}')
        self.assertEqual(cm.exception.context["field"], "columns[0].type")

    def test_bitmap_length_mismatch_rejected(self):
        payload = ('{"format":"b2","block_id":"blk-z","row_count":16,'
                   '"columns":[{"name":"a","type":"int64",'
                   '"null_bitmap":"00","values":['
                   + ",".join(["0"] * 16) + "]}]}")
        with self.assertRaises(FormatError) as cm:
            decode_block(payload)
        self.assertEqual(cm.exception.context["field"], "columns[0].null_bitmap")

    def test_unsupported_type_rejected(self):
        with self.assertRaises(FormatError) as cm:
            decode_block('{"format":"b2","block_id":"blk-w","row_count":1,'
                         '"columns":[{"name":"a","type":"string",'
                         '"null_bitmap":"00","values":[1]}]}')
        self.assertEqual(cm.exception.context["field"], "columns[0].type")

    def test_b1_null_rejected(self):
        with self.assertRaises(FormatError):
            decode_block('{"format":"b1","block_id":"blk-n","row_count":1,'
                         '"columns":[{"name":"a","values":[null]}]}')


if __name__ == "__main__":
    unittest.main()
