import struct
import tempfile
import unittest
from pathlib import Path

from pcl.digest import digest_bytes
from pcl.entries import (
    CURRENT_ENTRY_VERSION,
    Entry,
    decode_entry,
    encode_entry,
    normalize_to_v3,
)
from pcl.errors import CorruptEntryError, DigestMismatchError


def make_entry(fmt):
    return Entry(
        key="api-image",
        digest="sha256:" + "a" * 64,
        version=3,
        size=120,
        stage_id="stage-package",
        refs=(("core-bundle", "sha256:" + "b" * 64, 2),),
        compressed=False,
    )


class EntryCodecTests(unittest.TestCase):
    def test_round_trip_each_version(self):
        for fmt in (1, 2, 3):
            blob = encode_entry(make_entry(fmt), fmt)
            entry = decode_entry(blob, "api-image")
            self.assertEqual(entry.format_version, fmt)
            self.assertEqual(entry.digest, "sha256:" + "a" * 64)
            if fmt == 1:
                self.assertIsNone(entry.size)
                self.assertEqual(entry.refs, ())
            if fmt >= 2:
                self.assertEqual(entry.size, 120)
                self.assertEqual(entry.stage_id, "stage-package")
            if fmt == 3:
                self.assertEqual(len(entry.refs), 1)
                self.assertFalse(entry.compressed)

    def test_new_write_uses_current_version(self):
        blob = encode_entry(make_entry(3))
        self.assertEqual(struct.unpack(">B", blob[4:5])[0], CURRENT_ENTRY_VERSION)

    def test_length_self_check_truncation(self):
        blob = bytearray(encode_entry(make_entry(2), 2))
        blob = blob[: len(blob) - 10]
        with self.assertRaises(CorruptEntryError) as ctx:
            decode_entry(bytes(blob), "api-image")
        self.assertEqual(ctx.exception.code, "E_CORRUPT")
        self.assertEqual(ctx.exception.details["key"], "api-image")
        self.assertEqual(ctx.exception.details["offset"], 9)

    def test_bad_magic_offset_zero(self):
        blob = bytearray(encode_entry(make_entry(1), 1))
        blob[0] = ord("X")
        with self.assertRaises(CorruptEntryError) as ctx:
            decode_entry(bytes(blob), "api-image")
        self.assertEqual(ctx.exception.details["offset"], 0)

    def test_payload_flip_detected_by_digest(self):
        blob = bytearray(encode_entry(make_entry(3), 3))
        # Flip the final hex character of the payload content itself.
        payload_len = struct.unpack(">I", blob[5:9])[0]
        payload_end = 9 + payload_len
        blob[payload_end - 1] = ord("0") if blob[payload_end - 1] != 48 else ord("1")
        with self.assertRaises(CorruptEntryError) as ctx:
            decode_entry(bytes(blob), "api-image")
        self.assertEqual(
            ctx.exception.details["offset"],
            payload_end + 8,
        )

    def test_unsupported_version(self):
        blob = bytearray(encode_entry(make_entry(1), 1))
        blob[4] = 9
        with self.assertRaises(CorruptEntryError):
            decode_entry(bytes(blob), "api-image")

    def test_normalize_backfilled_fields(self):
        v1 = decode_entry(encode_entry(make_entry(1), 1), "api-image")
        n1 = normalize_to_v3(v1, size=120, stage_id="stage-package")
        self.assertEqual(n1.format_version, 3)
        self.assertEqual(
            set(n1.backfilled), {"compressed", "refs", "size", "stage_id"}
        )
        self.assertEqual(n1.refs, ())
        self.assertFalse(n1.compressed)

        v2 = decode_entry(encode_entry(make_entry(2), 2), "api-image")
        n2 = normalize_to_v3(v2)
        self.assertEqual(set(n2.backfilled), {"compressed", "refs"})

    def test_digest_mismatch_error_carries_both_sides(self):
        entry = make_entry(3)
        with self.assertRaises(DigestMismatchError) as ctx:
            from pcl.entries import verify_digest

            verify_digest(entry, b"different bytes", key="api-image")
        self.assertEqual(ctx.exception.code, "E_DIGEST")
        self.assertEqual(
            ctx.exception.details["expected_digest"], entry.digest
        )
        self.assertEqual(
            ctx.exception.details["actual_digest"],
            digest_bytes(b"different bytes"),
        )


if __name__ == "__main__":
    unittest.main()
