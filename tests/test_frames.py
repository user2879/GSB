from __future__ import annotations

import struct
import unittest

from h2burst.errors import WorkbenchError
from h2burst.frames import (
    FRAME_DATA,
    FRAME_GOAWAY,
    FRAME_HEADERS,
    FRAME_RST_STREAM,
    FRAME_SETTINGS,
    FRAME_WINDOW_UPDATE,
    STATIC_TABLE,
    Frame,
    decode_frame,
    decode_frames,
    decode_header_block,
    decode_settings,
    encode_frame,
    encode_frames,
    encode_header_block,
    make_data,
    make_goaway,
    make_headers,
    make_rst_stream,
    make_settings,
    make_window_update,
    validate_frame,
)


class FrameCodecTests(unittest.TestCase):
    def round_trip(self, frame: Frame) -> None:
        wire = encode_frame(frame)
        decoded = decode_frame(wire)
        self.assertEqual(decoded, frame)
        self.assertEqual(encode_frame(decoded), wire)

    def test_static_table_has_explicit_rfc_order(self) -> None:
        self.assertEqual(len(STATIC_TABLE), 61)
        self.assertEqual(STATIC_TABLE[0], (":authority", ""))
        self.assertEqual(STATIC_TABLE[1], (":method", "GET"))
        self.assertEqual(STATIC_TABLE[-1], ("www-authenticate", ""))

    def test_header_block_round_trip_and_static_order(self) -> None:
        headers = ((":scheme", "http"), ("content-type", "application/json"), (":method", "POST"))
        block = encode_header_block(headers)
        decoded = decode_header_block(block)
        self.assertEqual(
            decoded,
            ((":method", "POST"), (":scheme", "http"), ("content-type", "application/json")),
        )
        self.assertEqual(encode_header_block(decoded), block)

    def test_header_mapping_order_does_not_affect_bytes(self) -> None:
        first = {"user-agent": "local", ":path": "/x", "content-length": "0"}
        second = {"content-length": "0", ":path": "/x", "user-agent": "local"}
        self.assertEqual(encode_header_block(first), encode_header_block(second))

    def test_header_unknown_name_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            encode_header_block((("x-private", "value"),), frame_seq=4)
        self.assertEqual(ctx.exception.code, "E_FRAME")
        self.assertEqual(ctx.exception.frame_seq, 4)
        self.assertIn("headers[0].name", ctx.exception.field)

    def test_data_round_trip(self) -> None:
        self.round_trip(make_data(3, b"abcdef", end_stream=True))

    def test_headers_round_trip(self) -> None:
        self.round_trip(make_headers(1, {":method": "GET", ":path": "/"}, end_stream=True))

    def test_settings_round_trip_uses_sorted_identifiers(self) -> None:
        frame = make_settings({4: 65535, 2: 1, 5: 16384})
        self.round_trip(frame)
        self.assertEqual(decode_settings(frame.payload), ((2, 1), (4, 65535), (5, 16384)))

    def test_window_update_round_trip(self) -> None:
        self.round_trip(make_window_update(1, 1234))
        self.round_trip(make_window_update(0, 1))

    def test_rst_stream_round_trip(self) -> None:
        self.round_trip(make_rst_stream(5, 8))

    def test_goaway_round_trip(self) -> None:
        self.round_trip(make_goaway(43, 0, b"local"))

    def test_frame_sequence_round_trip(self) -> None:
        frames = [make_settings({4: 65535}), make_headers(1, {":method": "GET"}), make_data(1, b"x")]
        wire = encode_frames(frames)
        self.assertEqual(decode_frames(wire), tuple(frames))

    def test_declared_length_mismatch_is_e_frame(self) -> None:
        wire = encode_frame(make_data(1, b"abc"))
        bad = wire[:3] + struct.pack(">I", 10)[1:] + wire[3:]
        with self.assertRaises(WorkbenchError) as ctx:
            decode_frame(bad, frame_seq=9)
        self.assertEqual(ctx.exception.code, "E_FRAME")
        self.assertEqual(ctx.exception.field, "length")
        self.assertEqual(ctx.exception.frame_seq, 9)

    def test_unsupported_frame_type_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(99, 0, 0, b""), frame_seq=2)
        self.assertEqual(ctx.exception.code, "E_FRAME")
        self.assertEqual(ctx.exception.field, "frame_type")
        self.assertEqual(ctx.exception.frame_seq, 2)

    def test_connection_and_stream_level_ids_validated(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_SETTINGS, 0, 1, b""), frame_seq=1)
        self.assertEqual(ctx.exception.field, "stream_id")
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_DATA, 0, 0, b""), frame_seq=2)
        self.assertEqual(ctx.exception.field, "stream_id")

    def test_reserved_stream_bit_rejected(self) -> None:
        wire = struct.pack(">I", 0)[:3] + bytes((FRAME_DATA, 0)) + struct.pack(">I", 0x80000001) + b"x"
        with self.assertRaises(WorkbenchError) as ctx:
            decode_frame(wire, frame_seq=3)
        self.assertEqual(ctx.exception.code, "E_FRAME")
        self.assertEqual(ctx.exception.field, "stream_id")

    def test_settings_bad_length_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_SETTINGS, 0, 0, b"12345"), frame_seq=6)
        self.assertEqual(ctx.exception.field, "length")

    def test_settings_ack_payload_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_SETTINGS, 1, 0, struct.pack(">HI", 4, 1)), frame_seq=7)
        self.assertEqual(ctx.exception.field, "payload")

    def test_window_update_bad_length_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_WINDOW_UPDATE, 0, 1, b"123"), frame_seq=8)
        self.assertEqual(ctx.exception.field, "length")

    def test_rst_bad_length_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_RST_STREAM, 0, 1, b"12"), frame_seq=8)
        self.assertEqual(ctx.exception.field, "length")

    def test_goaway_bad_length_rejected(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_GOAWAY, 0, 0, b"123"), frame_seq=8)
        self.assertEqual(ctx.exception.field, "length")

    def test_headers_require_static_header_payload(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(Frame(FRAME_HEADERS, 0, 1, b"bad"), frame_seq=5)
        self.assertTrue(ctx.exception.field.startswith("headers"))

    def test_max_frame_size_enforced(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            validate_frame(make_data(1, b"x" * 11), frame_seq=4, max_frame_size=10)
        self.assertEqual(ctx.exception.field, "length")


if __name__ == "__main__":
    unittest.main()
