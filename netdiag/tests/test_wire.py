import binascii
import struct
import unittest
from netdiag.constants import FLAG_ACK, FLAG_RST, FLAG_SYN
from netdiag.errors import WireFormatError
from netdiag.wire import (compute_flow_tag, decode_buffer, encode_frame,
                          flow_key)

FLOW = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=40001,
            dst_port=50001, protocol=6)


def frame(seq, ts, flags=FLAG_ACK, payload=b"x"):
    return encode_frame(timestamp_us=ts, seq=seq, flags=flags,
                        payload=payload, **FLOW)


class WireTests(unittest.TestCase):
    def test_roundtrip_and_flow_reassembly(self):
        blob = frame(2, 2000) + frame(1, 1000)
        decoded = decode_buffer(blob)
        self.assertEqual(len(decoded.frames), 2)
        ordered = decoded.ordered_sequence()
        self.assertEqual([f.seq for f in ordered], [1, 2])
        self.assertEqual(decoded.frames[0].flow_key,
                         flow_key(**FLOW))

    def test_truncated_frame_reports_offset(self):
        blob = frame(1, 1000)
        with self.assertRaises(WireFormatError) as ctx:
            decode_buffer(blob[:-7])
        self.assertEqual(ctx.exception.code, "E_WIRE")
        self.assertIn("frame_offset", ctx.exception.context)

    def test_payload_length_out_of_range(self):
        big = b"z" * 1401
        with self.assertRaises(WireFormatError) as ctx:
            encode_frame(timestamp_us=1, seq=1, flags=FLAG_ACK,
                         payload=big, **FLOW)
        self.assertEqual(ctx.exception.code, "E_WIRE")

    def test_illegal_flag_combination(self):
        with self.assertRaises(WireFormatError):
            encode_frame(timestamp_us=1, seq=1,
                         flags=FLAG_SYN | FLAG_RST, **FLOW)

    def test_flow_tag_tamper_detected(self):
        blob = bytearray(frame(1, 1000))
        # 先重算 CRC 以绕过 CRC 校验，再篡改源端口，
        # 使五元组与帧内 flow_tag 不一致 -> 精准命中流标识校验。
        payload_len = struct.unpack("<H", blob[30:32])[0]
        body_len = 36 + payload_len
        blob[20] = 9
        crc = binascii.crc32(bytes(blob[:body_len])) & 0xFFFFFFFF
        blob[body_len:body_len + 4] = struct.pack("<I", crc)
        with self.assertRaises(WireFormatError) as ctx:
            decode_buffer(bytes(blob))
        self.assertEqual(ctx.exception.context["reason"],
                         "flow_tag_mismatch")
        self.assertEqual(ctx.exception.context["frame_offset"], 0)

    def test_direction_insensitive_tag(self):
        tag_a = compute_flow_tag("10.0.0.1", "10.0.0.2", 1, 2, 6)
        tag_b = compute_flow_tag("10.0.0.2", "10.0.0.1", 2, 1, 6)
        self.assertEqual(tag_a, tag_b)


if __name__ == "__main__":
    unittest.main()
