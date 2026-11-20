import unittest

from colbench import codecs
from colbench.errors import EncodingError, PayloadError


class CodecTests(unittest.TestCase):
    def test_all_encodings_roundtrip_int(self):
        values = [3, 3, 3, 8, 1, 1, 9, 9, 9, 9]
        for encoding in codecs.ENCODINGS:
            for compressed in (False, True):
                block = codecs.encode_block(values, encoding,
                                            compressed=compressed)
                decoded = codecs.decode_block(
                    block.payload, expect_encoding=encoding,
                    expect_digest=block.digest)
                self.assertEqual(decoded, values)

    def test_all_encodings_roundtrip_string(self):
        values = ["north", "north", "south", "east", "east", "east"]
        for encoding in codecs.ENCODINGS:
            block = codecs.encode_block(values, encoding, compressed=True)
            self.assertEqual(
                codecs.decode_block(block.payload,
                                    expect_encoding=encoding,
                                    expect_digest=block.digest), values)

    def test_deterministic_bytes_and_dict_order(self):
        values = ["zeta", "alpha", "alpha", "mid", "zeta"]
        first = codecs.encode_block(values, codecs.DICT, compressed=False)
        second = codecs.encode_block(list(reversed(values[::-1])),
                                     codecs.DICT, compressed=False)
        self.assertEqual(first.payload, second.payload)
        self.assertEqual(codecs.block_summary(values),
                         codecs.block_summary(list(values)))
        # 字典项必须按 UTF-8 字节序显式排序
        self.assertIn(b"alpha", first.payload)
        self.assertLess(first.payload.index(b"alpha"),
                        first.payload.index(b"mid"))
        self.assertLess(first.payload.index(b"mid"),
                        first.payload.index(b"zeta"))

    def test_truncation_is_payload_error(self):
        block = codecs.encode_block([1, 2, 3, 4], codecs.RAW,
                                    compressed=False)
        with self.assertRaises(PayloadError):
            codecs.decode_block(block.payload, truncate=len(block.payload) - 4)

    def test_digest_mismatch_is_payload_error(self):
        block = codecs.encode_block([1, 2, 3], codecs.RAW, compressed=False)
        with self.assertRaises(PayloadError):
            codecs.decode_block(block.payload,
                                expect_digest="0" * 64)

    def test_encoding_mismatch_is_encoding_error(self):
        block = codecs.encode_block([1, 1, 2], codecs.RLE, compressed=False)
        with self.assertRaises(EncodingError):
            codecs.decode_block(block.payload,
                                expect_encoding=codecs.RAW)


if __name__ == "__main__":
    unittest.main()
