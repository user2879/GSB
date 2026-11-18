from __future__ import annotations

import struct
import tempfile
from pathlib import Path
import unittest
import zlib

from pagecache.errors import EngineError, ErrorCode
from pagecache.manifest import decode_manifest, read_manifest, write_manifest
from pagecache.pagefile import read_page_file
from pagecache.segment import decode_segment, encode_segment, read_segment_file
from tests.helpers import make_samples, version_paths


class ManifestCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.data, cls.meta = make_samples(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_all_versions_read_and_current_write_is_deterministic(self) -> None:
        manifests = {}
        for version in (1, 2, 3):
            manifest_path, _pages, _overflow = version_paths(self.data, version)
            manifests[version] = read_manifest(manifest_path)
            self.assertEqual(manifests[version].header_version, version)
            self.assertEqual(manifests[version].format_versions, (version,))
            self.assertEqual(manifests[version].page_count, 8)
        self.assertIsNone(manifests[1].records[0].payload_checksum)
        self.assertEqual(manifests[1].records[0].column_summaries, ())
        self.assertIsNotNone(manifests[2].records[0].payload_checksum)
        self.assertGreater(len(manifests[2].records[0].column_summaries), 0)
        self.assertTrue(manifests[3].records[0].compressed)
        self.assertEqual(manifests[3].records[0].overflow_ref, "p0001.seg")

        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "a.manifest"
            second = Path(temporary) / "b.manifest"
            write_manifest(first, manifests[3].records, format_version=3)
            write_manifest(second, manifests[3].records, format_version=3)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(read_manifest(first), manifests[3])

    def test_tail_truncation_is_e_format_with_path_and_record_index(self) -> None:
        manifest_path, _pages, _overflow = version_paths(self.data, 3)
        data = manifest_path.read_bytes()[:-7]
        with self.assertRaises(EngineError) as caught:
            decode_manifest(data, source_path="tail.manifest")
        error = caught.exception
        self.assertIs(error.code, ErrorCode.E_FORMAT)
        self.assertIn(".records[", error.path or "")
        self.assertIsNotNone(error.record_index)

    def test_record_checksum_and_unsupported_versions_are_rejected(self) -> None:
        manifest_path, _pages, _overflow = version_paths(self.data, 3)
        data = bytearray(manifest_path.read_bytes())
        first_record_data_offset = 16 + 4
        data[first_record_data_offset + 5] ^= 0x01
        with self.assertRaises(EngineError) as caught:
            decode_manifest(bytes(data), source_path="crc.manifest")
        self.assertIs(caught.exception.code, ErrorCode.E_FORMAT)
        self.assertTrue((caught.exception.path or "").endswith("record_crc"))
        self.assertEqual(caught.exception.record_index, 0)

        unsupported = bytearray(manifest_path.read_bytes())
        struct.pack_into("<H", unsupported, 8, 9)
        with self.assertRaises(EngineError) as caught_version:
            decode_manifest(bytes(unsupported), source_path="version.manifest")
        self.assertIs(caught_version.exception.code, ErrorCode.E_FORMAT)
        self.assertTrue((caught_version.exception.path or "").endswith(".version"))

    def test_page_and_segment_compression_roundtrip(self) -> None:
        _manifest, pages_dir, overflow = version_paths(self.data, 3)
        page = read_page_file(pages_dir / "p0001.bin")
        compressed = read_segment_file(overflow / "p0001.seg")
        uncompressed = read_segment_file(version_paths(self.data, 2)[2] / "p0001.seg")
        self.assertTrue(compressed.compressed)
        self.assertFalse(uncompressed.compressed)
        self.assertEqual(compressed.payload, page.raw_bytes)
        self.assertEqual(uncompressed.payload, page.raw_bytes)
        self.assertEqual(compressed.page(), page)

    def test_compression_flag_mismatch_reports_page_id(self) -> None:
        _manifest, pages_dir, _overflow = version_paths(self.data, 1)
        page = read_page_file(pages_dir / "p0002.bin")
        encoded = bytearray(
            encode_segment(page_id="p0002", row_count=page.row_count, payload=page.raw_bytes, compressed=False, version=3)
        )
        encoded[10] = 1  # flags: claim compressed although the payload is plain
        struct.pack_into("<I", encoded, len(encoded) - 4, zlib.crc32(encoded[:-12]) & 0xFFFFFFFF)
        with self.assertRaises(EngineError) as caught:
            decode_segment(bytes(encoded), source_path="mismatch.seg")
        self.assertIs(caught.exception.code, ErrorCode.E_PAYLOAD)
        self.assertEqual(caught.exception.page_id, "p0002")


if __name__ == "__main__":
    unittest.main()
