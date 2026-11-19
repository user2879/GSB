from __future__ import annotations

import unittest

from h2mux.errors import FrameError
from h2mux.frames import (
    Frame,
    HeaderField,
    PrioritySpec,
    Setting,
    decode_frame,
    encode_frame,
)
from h2mux.hpack import (
    STATIC_TABLE_COUNT,
    decode_header_block,
    encode_header_block,
)


class FrameCodecTests(unittest.TestCase):
    def assert_round_trip(self, frame: Frame) -> None:
        raw = encode_frame(frame, preserve_raw=False)
        decoded = decode_frame(raw)
        self.assertEqual(encode_frame(decoded), raw)
        self.assertEqual(decoded.type, frame.type)
        self.assertEqual(decoded.stream_id, frame.stream_id)

    def test_supported_frame_round_trips(self) -> None:
        frames = [
            Frame.settings_frame([Setting(3, 3), Setting(4, 18000)]),
            Frame.settings_frame(ack=True),
            Frame.headers_frame(
                1,
                [
                    HeaderField.indexed(2),
                    HeaderField.literal_indexed_name(4, b"/stream/1"),
                    HeaderField.literal(b"x-test", b"yes"),
                ],
                end_stream=True,
                priority=PrioritySpec(depends_on=0, weight=32, exclusive=True),
            ),
            Frame.data_frame(3, b"payload", end_stream=True),
            Frame.window_update_frame(0, 30000),
            Frame.window_update_frame(5, 14000),
            Frame.rst_stream_frame(7, 8),
            Frame.goaway_frame(9, 0, b"done"),
        ]
        for frame in frames:
            with self.subTest(frame=frame.type):
                self.assert_round_trip(frame)

    def test_hpack_static_and_literal_round_trip(self) -> None:
        fields = (
            HeaderField.indexed(1),
            HeaderField.indexed(2),
            HeaderField.literal_indexed_name(4, b"/offline"),
            HeaderField.literal(b"x-local", b"yes"),
        )
        block = encode_header_block(fields)
        self.assertEqual(decode_header_block(block), fields)
        self.assertEqual(encode_header_block(decode_header_block(block)), block)

    def test_static_index_out_of_range_reports_offset(self) -> None:
        invalid_index = STATIC_TABLE_COUNT + 1
        payload = bytes([0x80 | invalid_index])
        raw = len(payload).to_bytes(3, "big") + bytes([0x1, 0x4]) + (1).to_bytes(4, "big") + payload
        with self.assertRaises(FrameError) as context:
            decode_frame(raw)
        self.assertEqual(context.exception.code, "E_FRAME")
        self.assertEqual(context.exception.frame_type, "HEADERS")
        self.assertEqual(context.exception.offset, 9)

    def test_even_stream_id_rejected(self) -> None:
        raw = (0).to_bytes(3, "big") + bytes([0x0, 0x0]) + (2).to_bytes(4, "big")
        with self.assertRaises(FrameError) as context:
            decode_frame(raw)
        self.assertEqual(context.exception.frame_type, "DATA")
        self.assertEqual(context.exception.offset, 5)

    def test_connection_frame_requires_stream_zero(self) -> None:
        raw = (0).to_bytes(3, "big") + bytes([0x4, 0x0]) + (1).to_bytes(4, "big")
        with self.assertRaises(FrameError) as context:
            decode_frame(raw)
        self.assertEqual(context.exception.frame_type, "SETTINGS")
        self.assertEqual(context.exception.offset, 5)

    def test_payload_over_configured_limit(self) -> None:
        raw = (20).to_bytes(3, "big") + bytes([0x0, 0x0]) + (1).to_bytes(4, "big") + b"x" * 20
        with self.assertRaises(FrameError) as context:
            decode_frame(raw, max_frame_size=16)
        self.assertEqual(context.exception.frame_type, "DATA")
        self.assertEqual(context.exception.offset, 0)

    def test_truncated_payload_reports_end_offset(self) -> None:
        raw = (10).to_bytes(3, "big") + bytes([0x0, 0x0]) + (1).to_bytes(4, "big") + b"abc"
        with self.assertRaises(FrameError) as context:
            decode_frame(raw)
        self.assertEqual(context.exception.offset, len(raw))

    def test_settings_ack_payload_rejected(self) -> None:
        raw = (6).to_bytes(3, "big") + bytes([0x4, 0x1]) + (0).to_bytes(4, "big") + b"\x00" * 6
        with self.assertRaises(FrameError) as context:
            decode_frame(raw)
        self.assertEqual(context.exception.frame_type, "SETTINGS")

    def test_encoder_rejects_zero_window_increment(self) -> None:
        with self.assertRaises(FrameError) as context:
            encode_frame(Frame.window_update_frame(1, 0), preserve_raw=False)
        self.assertEqual(context.exception.frame_type, "WINDOW_UPDATE")
        self.assertEqual(context.exception.offset, 9)


if __name__ == "__main__":
    unittest.main()
