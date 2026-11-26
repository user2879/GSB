import unittest

from gsb.errors import BackfillError, FormatError
from gsb.receipts import (
    X1_LINE_WIDTH,
    canonical_normalized_text,
    decode,
    encode_x1,
    encode_x2,
    encode_x3,
    normalize,
)

BATCH = "a" * 32
RECORD = {
    "dispatch_id": "disp-1",
    "subscriber_id": "sub-alpha",
    "event_seqs": [1, 3, 5],
    "digest": "d" * 64,
    "completion_ms": 4321,
    "caliber_version": "v1",
    "covered_refs": [],
}


class ReceiptCodecTest(unittest.TestCase):
    def test_x1_fixed_width_roundtrip(self):
        line = encode_x1(BATCH, -12345)
        self.assertEqual(len(line), X1_LINE_WIDTH)
        decoded = decode(line)
        self.assertEqual(decoded["format_version"], "x1")
        self.assertEqual(decoded["fixed_point_sum"], -12345)

    def test_x1_short_line_rejected(self):
        with self.assertRaises(FormatError) as ctx:
            decode(BATCH + "123")
        self.assertEqual(ctx.exception.code, "E_FORMAT")

    def test_unknown_version_rejected(self):
        with self.assertRaises(FormatError) as ctx:
            decode("x9|a|b")
        self.assertEqual(ctx.exception.code, "E_FORMAT")

    def test_x2_x3_roundtrip(self):
        x2 = decode(encode_x2(BATCH, "sub-alpha", 4321, 999))
        self.assertEqual(x2["completion_ms"], 4321)
        x3 = decode(encode_x3(BATCH, "sub-alpha", 4321, 999, [5, 1, 3], [2], "v1"))
        self.assertEqual(x3["event_seqs"], [5, 1, 3])

    def test_normalization_backfills_and_is_byte_identical(self):
        x1 = normalize(encode_x1(BATCH, 999), lambda _id: RECORD)
        x2 = normalize(encode_x2(BATCH, "sub-alpha", 4321, 999), lambda _id: RECORD)
        x3 = normalize(
            encode_x3(BATCH, "sub-alpha", 4321, 999, [1, 3, 5], [], "v1"),
            lambda _id: RECORD,
        )
        self.assertEqual(
            x1["backfilled_fields"],
            [
                "caliber_version",
                "completion_ms",
                "covered_refs",
                "event_seqs",
                "subscriber_id",
            ],
        )
        self.assertTrue(x1["backfill_basis"].startswith("ledger:dispatch:disp-1"))
        self.assertEqual(
            x2["backfilled_fields"], ["caliber_version", "covered_refs", "event_seqs"]
        )
        self.assertEqual(x3["backfilled_fields"], [])
        self.assertEqual(canonical_normalized_text(x1), canonical_normalized_text(x2))
        self.assertEqual(canonical_normalized_text(x2), canonical_normalized_text(x3))

    def test_untraceable_backfill_rejected(self):
        with self.assertRaises(BackfillError) as ctx:
            normalize(encode_x1(BATCH, 1), lambda _id: None)
        self.assertEqual(ctx.exception.code, "E_BACKFILL")


if __name__ == "__main__":
    unittest.main()
