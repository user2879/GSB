import unittest

from convd import wire
from convd.errors import WireError


class TestWire(unittest.TestCase):
    def test_pulse_roundtrip(self):
        buf = wire.encode_pulse("sc", 3, 250, "tgt", 123456)
        kind, msg = wire.decode_datagram(buf)
        self.assertEqual(kind, "pulse")
        self.assertEqual(msg, {"scenario_id": "sc", "round": 3,
                               "lease_ms": 250, "target_digest": "tgt",
                               "send_mono_ms": 123456})

    def test_receipt_roundtrip(self):
        buf = wire.encode_receipt("exe-a", 2, 2, 999, "d" * 64)
        kind, msg = wire.decode_datagram(buf)
        self.assertEqual(kind, "receipt")
        self.assertEqual(msg["obs_seq"], 2)
        self.assertEqual(msg["obs_mono_ms"], 999)

    def test_error_roundtrip(self):
        buf = wire.encode_error("exe-a", 1, "E_STALE")
        kind, msg = wire.decode_datagram(buf)
        self.assertEqual(kind, "error")
        self.assertEqual(msg["code"], "E_STALE")

    def test_truncated_reports_offset(self):
        buf = wire.encode_pulse("sc", 3, 250, "tgt", 123456)
        cut = buf[:-3]
        with self.assertRaises(WireError) as ctx:
            wire.decode_datagram(cut)
        self.assertEqual(ctx.exception.code, "E_WIRE")
        self.assertGreaterEqual(ctx.exception.offset, 0)
        self.assertLess(ctx.exception.offset, len(buf))

    def test_trailing_bytes_rejected(self):
        buf = wire.encode_pulse("sc", 3, 250, "tgt", 1) + b"xx"
        with self.assertRaises(WireError) as ctx:
            wire.decode_datagram(buf)
        self.assertEqual(ctx.exception.offset, len(buf) - 2)

    def test_bad_magic(self):
        with self.assertRaises(WireError) as ctx:
            wire.decode_datagram(b"XXXX1234")
        self.assertEqual(ctx.exception.offset, 0)

    def test_round_zero_out_of_range(self):
        buf = wire.encode_pulse("sc", 0, 250, "tgt", 1)
        with self.assertRaises(WireError):
            wire.decode_datagram(buf)

    def test_obs_seq_regression_detected(self):
        msg = {"executor_id": "e", "round": 2, "obs_seq": 1,
               "obs_mono_ms": 1, "state_digest": "d"}
        with self.assertRaises(WireError):
            wire.validate_receipt(msg, "sc", 2, {"e": 2})

    def test_round_mismatch_detected(self):
        msg = {"executor_id": "e", "round": 3, "obs_seq": 3,
               "obs_mono_ms": 1, "state_digest": "d"}
        with self.assertRaises(WireError):
            wire.validate_receipt(msg, "sc", 2, {})


if __name__ == "__main__":
    unittest.main()
