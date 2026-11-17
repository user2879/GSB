from __future__ import annotations

import copy
import threading
import unittest

from tunnelvpn.budget import ManualClock
from tunnelvpn.codec import (
    FLAG_FRAGMENTED,
    FLAG_LAST_FRAGMENT,
    HEADER_SIZE,
    FrameDecoder,
    Reassembler,
    SequenceAllocator,
    decode_frame,
    encode_message,
    payload_digest_hex,
)
from tunnelvpn.errors import ErrorCode, TunnelError


class CodecTests(unittest.TestCase):
    def test_deterministic_encoding(self) -> None:
        kwargs = dict(
            tunnel_id="alpha",
            sequence=7,
            cipher="AES-256-GCM",
            mtu=192,
            payload=(b"deterministic-payload-" * 10),
        )
        first = encode_message(**kwargs)
        second = encode_message(**copy.deepcopy(kwargs))
        self.assertEqual(first.frames, second.frames)
        self.assertEqual(first.payload_digest, payload_digest_hex(kwargs["payload"]))
        self.assertTrue(all(len(frame) <= 192 for frame in first.frames))

    def test_unfragmented_roundtrip(self) -> None:
        payload = b"short payload"
        message = encode_message(
            tunnel_id="alpha", sequence=0, cipher="NONE", mtu=192, payload=payload
        )
        self.assertFalse(message.fragmented)
        decoded = decode_frame(message.frames[0], expected_mtu=192)
        self.assertEqual(decoded.payload, payload)
        self.assertFalse(decoded.fragmented)
        self.assertEqual(decoded.sequence, 0)

    def test_fragment_and_force_fragment_have_same_logical_digest(self) -> None:
        payload = b"F" * 321
        automatic = encode_message(
            tunnel_id="alpha", sequence=1, cipher="NONE", mtu=128, payload=payload
        )
        forced = encode_message(
            tunnel_id="alpha",
            sequence=1,
            cipher="NONE",
            mtu=9000,
            payload=payload,
            force_fragment=True,
        )
        self.assertTrue(automatic.fragmented)
        self.assertTrue(forced.fragmented)
        self.assertEqual(automatic.payload_digest, forced.payload_digest)
        for message in (automatic, forced):
            reassembler = Reassembler(timeout_ms=1000)
            decoder = FrameDecoder()
            result = None
            for frame in reversed(message.frames):
                result = reassembler.add(decoder.decode(frame))
            self.assertIsNotNone(result)
            self.assertEqual(result.payload, payload)
            self.assertEqual(result.payload_digest, message.payload_digest)

    def test_out_of_order_and_exact_duplicate(self) -> None:
        payload = b"out-of-order-" * 40
        message = encode_message(
            tunnel_id="alpha", sequence=4, cipher="NONE", mtu=128, payload=payload
        )
        self.assertGreaterEqual(len(message.frames), 3)
        reassembler = Reassembler(timeout_ms=1000)
        frames = [message.frames[2], message.frames[0], message.frames[0], message.frames[1]]
        frames.extend(message.frames[3:])
        result = None
        for frame in frames:
            result = reassembler.add(decode_frame(frame))
        self.assertIsNotNone(result)
        self.assertEqual(result.payload, payload)
        self.assertEqual(result.duplicates, 1)

    def test_missing_fragment_timeout_reports_offsets(self) -> None:
        clock = ManualClock(1_000_000_000)
        payload = b"M" * 300
        message = encode_message(
            tunnel_id="alpha", sequence=8, cipher="NONE", mtu=128, payload=payload
        )
        reassembler = Reassembler(timeout_ms=50, clock_ns=clock.now_ns)
        for index, frame in enumerate(message.frames):
            if index == 2:
                continue
            self.assertIsNone(reassembler.add(decode_frame(frame)))
        self.assertEqual(reassembler.sweep(), [])
        clock.advance_ms(51)
        errors = reassembler.sweep()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, ErrorCode.E_FRAG.value)
        self.assertEqual(errors[0].tunnel_id, "alpha")
        self.assertEqual(errors[0].details["sequence"], 8)
        self.assertEqual(errors[0].details["missing_offsets"], [88])

    def test_conflicting_duplicate_is_encap_error(self) -> None:
        payload = b"C" * 250
        message = encode_message(
            tunnel_id="alpha", sequence=9, cipher="NONE", mtu=128, payload=payload
        )
        first = decode_frame(message.frames[0])
        conflicting = copy.deepcopy(first)
        object.__setattr__(conflicting, "payload", b"X" * len(first.payload))
        reassembler = Reassembler(timeout_ms=1000)
        reassembler.add(first)
        with self.assertRaises(TunnelError) as caught:
            reassembler.add(conflicting)
        self.assertEqual(caught.exception.code, "E_ENCAP")
        self.assertGreaterEqual(caught.exception.offset, HEADER_SIZE)

    def test_digest_corruption_reports_first_digest_byte(self) -> None:
        message = encode_message(
            tunnel_id="alpha", sequence=1, cipher="NONE", mtu=192, payload=b"payload"
        )
        corrupted = bytearray(message.frames[0])
        corrupted[HEADER_SIZE - 16] ^= 0x80
        with self.assertRaises(TunnelError) as caught:
            decode_frame(bytes(corrupted))
        self.assertEqual(caught.exception.code, "E_ENCAP")
        self.assertEqual(caught.exception.offset, HEADER_SIZE - 16)

    def test_length_field_inconsistency(self) -> None:
        message = encode_message(
            tunnel_id="alpha", sequence=1, cipher="NONE", mtu=192, payload=b"payload"
        )
        with self.assertRaises(TunnelError) as caught:
            decode_frame(message.frames[0] + b"extra")
        self.assertEqual(caught.exception.code, "E_ENCAP")
        self.assertEqual(caught.exception.offset, 48)

    def test_sequence_guard_rejects_lower_and_completed_duplicate(self) -> None:
        decoder = FrameDecoder()
        one = encode_message(
            tunnel_id="alpha", sequence=2, cipher="NONE", mtu=192, payload=b"one"
        )
        two = encode_message(
            tunnel_id="alpha", sequence=1, cipher="NONE", mtu=192, payload=b"two"
        )
        decoder.decode(one.frames[0])
        decoder.mark_complete("alpha", 2)
        with self.assertRaises(TunnelError) as caught:
            decoder.decode(two.frames[0])
        self.assertEqual(caught.exception.offset, 40)
        with self.assertRaises(TunnelError):
            decoder.decode(one.frames[0])

    def test_same_sequence_allowed_for_pending_fragments(self) -> None:
        message = encode_message(
            tunnel_id="alpha", sequence=3, cipher="NONE", mtu=128, payload=b"Z" * 200
        )
        decoder = FrameDecoder()
        for frame in reversed(message.frames):
            decoded = decoder.decode(frame)
            self.assertEqual(decoded.sequence, 3)

    def test_sequence_allocator_is_thread_safe(self) -> None:
        allocator = SequenceAllocator()
        values: list[int] = []
        lock = threading.Lock()

        def allocate() -> None:
            for _ in range(200):
                value = allocator.allocate("alpha")
                with lock:
                    values.append(value)

        threads = [threading.Thread(target=allocate) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(values), 1600)
        self.assertEqual(sorted(values), list(range(1600)))
        self.assertEqual(allocator.watermark("alpha"), 1599)

    def test_fragment_header_flags_are_consistent(self) -> None:
        payload = b"flags-" * 80
        message = encode_message(
            tunnel_id="alpha", sequence=1, cipher="NONE", mtu=128, payload=payload
        )
        decoded = [decode_frame(frame) for frame in message.frames]
        self.assertTrue(all(item.flags & FLAG_FRAGMENTED for item in decoded))
        self.assertFalse(decoded[0].flags & FLAG_LAST_FRAGMENT)
        self.assertTrue(decoded[-1].flags & FLAG_LAST_FRAGMENT)


if __name__ == "__main__":
    unittest.main()
