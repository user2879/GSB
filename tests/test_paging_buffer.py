import unittest
from pathlib import Path
import tempfile

from dwb_diag.config import DEFAULT_PAGE_SIZE
from dwb_diag.buffer import DoubleWriteBuffer
from dwb_diag.errors import PageFormatError
from dwb_diag.paging import MainDataFile, Page


class TestPaging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.page = Page(DEFAULT_PAGE_SIZE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frame_roundtrip_and_digest(self):
        payload = b'{"v":1,"entries":{"a":1}}'
        frame = self.page.encode(payload)
        self.assertEqual(len(frame), DEFAULT_PAGE_SIZE)
        self.assertTrue(Page.structural_valid(frame, DEFAULT_PAGE_SIZE))
        self.assertEqual(self.page.decode_payload(frame), payload)
        self.assertEqual(len(Page.digest(frame)), 32)

    def test_unsupported_page_size(self):
        with self.assertRaises(ValueError):
            Page(333)

    def test_main_file_addressing(self):
        main = MainDataFile(self.root / "main.pages", DEFAULT_PAGE_SIZE, 8)
        main.ensure_file()
        self.assertEqual(main.path.stat().st_size, 8 * DEFAULT_PAGE_SIZE)
        frame = self.page.encode(b"hello")
        main.write_slot(3, frame)
        self.assertIsNone(main.read_slot(0))
        self.assertEqual(main.read_slot(3), frame)
        self.assertTrue(main.slot_intact(3, Page.digest(frame)))

    def test_page_no_range(self):
        main = MainDataFile(self.root / "m.pages", DEFAULT_PAGE_SIZE, 4)
        with self.assertRaises(PageFormatError):
            main.read_slot(4)


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dwb = DoubleWriteBuffer(Path(self.tmp.name) / "b.dwb", capacity=4)
        self.page = Page(DEFAULT_PAGE_SIZE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_strict_scan_ok(self):
        for seq, pno in enumerate([0, 1, 2], start=1):
            self.dwb.append_mirror(pno, DEFAULT_PAGE_SIZE, seq, self.page.encode(f"p{pno}".encode()))
        result = self.dwb.validate_strict()
        self.assertEqual([m.page_no for m in result.mirrors], [0, 1, 2])
        self.assertEqual(result.max_seq, 3)
        self.assertEqual(set(result.latest_per_page), {0, 1, 2})

    def test_seq_must_strictly_increase(self):
        self.dwb.append_mirror(0, DEFAULT_PAGE_SIZE, 1, self.page.encode(b"a"))
        self.dwb.append_mirror(1, DEFAULT_PAGE_SIZE, 1, self.page.encode(b"b"))
        with self.assertRaises(PageFormatError) as ctx:
            self.dwb.validate_strict()
        self.assertEqual(ctx.exception.code, "E_PAGE")
        self.assertEqual(ctx.exception.detail["reason"], "seq_not_increasing")
        self.assertIsNotNone(ctx.exception.detail["offset"])

    def test_page_no_out_of_range(self):
        with self.assertRaises(PageFormatError) as ctx:
            self.dwb.append_mirror(9, DEFAULT_PAGE_SIZE, 1, self.page.encode(b"a"))
        self.assertEqual(ctx.exception.code, "E_PAGE")

    def test_truncated_payload_is_e_page(self):
        self.dwb.append_mirror(0, DEFAULT_PAGE_SIZE, 1, self.page.encode(b"a"))
        path = self.dwb.path
        data = path.read_bytes()
        path.write_bytes(data[:-10])  # 截断尾部镜像
        with self.assertRaises(PageFormatError) as ctx:
            self.dwb.validate_strict()
        self.assertEqual(ctx.exception.code, "E_PAGE")
        self.assertIn("truncated", ctx.exception.detail["reason"])

    def test_digest_mismatch_strict_vs_lenient(self):
        self.dwb.append_mirror(0, DEFAULT_PAGE_SIZE, 1, self.page.encode(b"a"))
        path = self.dwb.path
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 0xFF  # 破坏镜像内容
        path.write_bytes(bytes(blob))
        with self.assertRaises(PageFormatError):
            self.dwb.validate_strict()
        result = self.dwb.scan(strict_digest=False)
        self.assertEqual(result.discarded_count, 1)
        self.assertEqual(result.usable, [])


if __name__ == "__main__":
    unittest.main()
