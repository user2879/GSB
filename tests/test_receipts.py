import unittest

from src import receipts
from src.errors import FieldError, RefError

D = "a" * 64


class TestReceiptCodec(unittest.TestCase):
    def test_roundtrip_all_formats(self):
        lines = [
            f"r1|batch-0000|req-0000|{D}",
            f"r2|batch-0000|req-0000|3|15|w0|{D}",
            f"r3|batch-0000|req-0000|3|15|w0|0||{D}",
            f"r3|batch-0000|req-0001|3|16|w0|1|req-0001|{D}",
        ]
        for line in lines:
            rec = receipts.parse_line(line, line_no=1)
            self.assertEqual(receipts.format_line(rec), line)

    def test_real_fields_tracked(self):
        rec = receipts.parse_line(f"r1|batch-0000|req-0000|{D}")
        self.assertTrue(rec.real("output_digest"))
        self.assertFalse(rec.real("param_version"))
        self.assertFalse(rec.real("rank"))

    def test_e_field_negative_elapsed(self):
        text = f"r2|batch-0000|req-0000|1|12|w0|{D}\nr2|batch-0000|req-0001|1|-5|w0|{D}\n"
        with self.assertRaises(FieldError) as ctx:
            receipts.parse_text(text)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertEqual(ctx.exception.loc["line"], 2)
        self.assertEqual(ctx.exception.loc["field"], "elapsed_ms")

    def test_e_field_digest_width(self):
        with self.assertRaises(FieldError) as ctx:
            receipts.parse_text("r1|batch-0000|req-0000|abc\n")
        self.assertEqual(ctx.exception.loc["field"], "output_digest")
        self.assertEqual(ctx.exception.loc["line"], 1)

    def test_e_field_id_width(self):
        with self.assertRaises(FieldError) as ctx:
            receipts.parse_text(f"r1|batch-000|req-0000|{D}\n")
        self.assertEqual(ctx.exception.loc["field"], "batch_id")

    def test_e_field_wrong_field_count(self):
        with self.assertRaises(FieldError) as ctx:
            receipts.parse_text(f"r1|batch-0000|req-0000|extra|{D}\n")
        self.assertEqual(ctx.exception.loc["field"], "format")

    def test_e_field_unknown_tag(self):
        with self.assertRaises(FieldError):
            receipts.parse_text(f"r9|batch-0000|req-0000|{D}\n")

    def test_e_ref_rank_duplicate(self):
        text = (f"r3|batch-0000|req-0000|1|1|w0|0||{D}\n"
                f"r3|batch-0000|req-0001|1|1|w0|0||{D}\n")
        with self.assertRaises(RefError) as ctx:
            receipts.parse_text(text)
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertEqual(ctx.exception.loc["line"], 2)
        self.assertEqual(ctx.exception.loc["field"], "rank")

    def test_e_ref_rank_out_of_range(self):
        text = (f"r3|batch-0000|req-0000|1|1|w0|0||{D}\n"
                f"r3|batch-0000|req-0001|1|1|w0|5||{D}\n")
        with self.assertRaises(RefError) as ctx:
            receipts.parse_text(text)
        self.assertEqual(ctx.exception.loc["field"], "rank")
        self.assertEqual(ctx.exception.loc["line"], 2)

    def test_e_ref_dangling_dup_of(self):
        text = (f"r3|batch-0000|req-0000|1|1|w0|0|req-9999|{D}\n"
                f"r3|batch-0000|req-0001|1|1|w0|1||{D}\n")
        with self.assertRaises(RefError) as ctx:
            receipts.parse_text(text)
        self.assertEqual(ctx.exception.loc["field"], "dup_of")
        self.assertEqual(ctx.exception.loc["line"], 1)

    def test_dup_of_self_reference_allowed(self):
        text = (f"r3|batch-0000|req-0000|1|1|w0|0||{D}\n"
                f"r3|batch-0000|req-0000|1|1|w0|1|req-0000|{D}\n")
        recs = receipts.parse_text(text)
        self.assertEqual(len(recs), 2)

    def test_first_illegal_receipt_reported(self):
        text = (f"r1|batch-0000|req-0000|{D}\n"
                f"r1|batch-0000|req-0001|{D}\n"
                "r1|batch-0000|req-0002|zz\n")
        with self.assertRaises(FieldError) as ctx:
            receipts.parse_text(text)
        self.assertEqual(ctx.exception.loc["line"], 3)


if __name__ == "__main__":
    unittest.main()
