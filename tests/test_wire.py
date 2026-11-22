import unittest

from dns_workbench.errors import WireError
from dns_workbench.wire import (
    CLASS_IN,
    TYPE_A,
    TYPE_CNAME,
    DnsMessage,
    Question,
    ResourceRecord,
    build_response_flags,
    encode_message,
    encode_tcp_frame,
    parse_message,
    truncate_for_udp,
)


class WireCodecTest(unittest.TestCase):
    def _response(self) -> DnsMessage:
        question = Question("www.example.test", TYPE_A)
        answers = [
            ResourceRecord("www.example.test", TYPE_CNAME, CLASS_IN, 300, b"", "edge.example.test"),
            ResourceRecord("edge.example.test", TYPE_A, CLASS_IN, 300, bytes([10, 0, 0, 5]), "10.0.0.5"),
        ]
        return DnsMessage(0x1234, build_response_flags(rcode=0, aa=True), [question], answers)

    def test_roundtrip_with_compression(self) -> None:
        message = self._response()
        data = encode_message(message)
        parsed = parse_message(data)
        self.assertEqual(parsed.txn_id, 0x1234)
        self.assertTrue(parsed.aa)
        self.assertEqual(parsed.questions[0].qname, "www.example.test")
        self.assertEqual(parsed.answers[1].rdata_text, "10.0.0.5")
        self.assertEqual(parsed.answers[0].rdata_text, "edge.example.test")
        self.assertLess(data.index(b"\xc0"), len(data))  # 确实使用了压缩指针

    def test_pointer_loop_rejected_with_offset(self) -> None:
        loop = bytes.fromhex("abcd81800001000000000000") + b"\x03www\xc0\x0c"
        with self.assertRaises(WireError) as ctx:
            parse_message(loop)
        self.assertEqual(ctx.exception.code, "E_WIRE")
        self.assertEqual(ctx.exception.offset, 16)

    def test_forward_pointer_rejected(self) -> None:
        forward = bytes.fromhex("abcd81800001000000000000") + b"\x03www\xc0\x10"
        with self.assertRaises(WireError):
            parse_message(forward)

    def test_pointer_out_of_bounds(self) -> None:
        oob = bytes.fromhex("abcd81800001000000000000") + b"\x03www\xc0\xff"
        with self.assertRaises(WireError) as ctx:
            parse_message(oob)
        self.assertEqual(ctx.exception.context["pointer"], 255)

    def test_short_header(self) -> None:
        with self.assertRaises(WireError):
            parse_message(b"\x00" * 5)

    def test_tail_bytes_rejected(self) -> None:
        data = encode_message(self._response()) + b"\x00"
        with self.assertRaises(WireError):
            parse_message(data)

    def test_udp_truncation_sets_tc_and_respects_limit(self) -> None:
        question = Question("big.example.test", TYPE_A)
        answers = [
            ResourceRecord("big.example.test", TYPE_A, CLASS_IN, 60, bytes([10, 1, 1, i % 256]), f"10.1.1.{i % 256}")
            for i in range(40)
        ]
        message = DnsMessage(1, build_response_flags(rcode=0, aa=True), [question], answers)
        truncated, tc = truncate_for_udp(message)
        self.assertTrue(tc)
        self.assertLessEqual(len(truncated), 512)
        self.assertTrue(parse_message(truncated).tc)

    def test_tcp_frame_roundtrip(self) -> None:
        frame = encode_tcp_frame(self._response())
        import io

        class FakeSocket:
            def __init__(self, payload: bytes) -> None:
                self._buffer = io.BytesIO(payload)

            def recv(self, size: int) -> bytes:
                return self._buffer.read(size)

        from dns_workbench.wire import read_tcp_frame

        payload = read_tcp_frame(FakeSocket(frame))  # type: ignore[arg-type]
        self.assertEqual(parse_message(payload).txn_id, 0x1234)


if __name__ == "__main__":
    unittest.main()
