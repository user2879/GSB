import unittest

from convd import wire
from convd.executor import Executor, advance_ms_for, state_digest_for


def make():
    return Executor("sc", "exe-a", 41001, 7, "tgt", base_ms=0, jitter_ms=0)


def pulse(round_no, send=1000):
    kind, msg = wire.decode_datagram(
        wire.encode_pulse("sc", round_no, 250, "tgt", send))
    return msg


class TestExecutor(unittest.TestCase):
    def test_version_advances_to_round(self):
        ex = make()
        reply = ex.handle_pulse(pulse(1))
        kind, msg = wire.decode_datagram(reply)
        self.assertEqual(kind, "receipt")
        self.assertEqual(msg["obs_seq"], 1)
        self.assertEqual(ex.version, 1)
        self.assertEqual(msg["state_digest"],
                         state_digest_for("sc", "exe-a", 1, "tgt"))

    def test_duplicate_pulse_is_idempotent(self):
        ex = make()
        first = ex.handle_pulse(pulse(1))
        second = ex.handle_pulse(pulse(1, send=2000))
        self.assertEqual(first, second)  # byte-identical replay
        self.assertEqual(ex.version, 1)  # no double advancement

    def test_stale_round_rejected(self):
        ex = make()
        ex.handle_pulse(pulse(2))
        reply = ex.handle_pulse(pulse(1))
        kind, msg = wire.decode_datagram(reply)
        self.assertEqual(kind, "error")
        self.assertEqual(msg["code"], "E_STALE")

    def test_deterministic_advance(self):
        self.assertEqual(advance_ms_for(11, 3, 1, 4),
                         advance_ms_for(11, 3, 1, 4))
        self.assertNotEqual(advance_ms_for(11, 3, 1, 4),
                            advance_ms_for(12, 3, 1, 4))

    def test_deterministic_trajectory(self):
        def run():
            ex = Executor("sc", "e", 1, 42, "tgt", base_ms=0, jitter_ms=0)
            out = []
            for r in (1, 2, 3):
                out.append(ex.handle_pulse(pulse(r)))
            return out
        self.assertEqual(run(), run())

    def test_drop_first_attempt(self):
        ex = Executor("sc", "e", 1, 5, "tgt", drop_first_attempt_rounds={2})
        ex.handle_pulse(pulse(1))
        self.assertIsNone(ex.handle_pulse(pulse(2)))   # dropped
        self.assertIsNotNone(ex.handle_pulse(pulse(2)))  # retry replies
        self.assertEqual(ex.version, 2)


if __name__ == "__main__":
    unittest.main()
