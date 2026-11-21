import unittest

from delayqueue import IllegalTransitionError
from delayqueue.models import (
    ACKED, DEAD, DELAYED, INFLIGHT, PENDING, READY, next_state,
)


class StateMachineTest(unittest.TestCase):
    def test_allowed_paths(self):
        self.assertEqual(next_state(PENDING, "enqueue"), DELAYED)
        self.assertEqual(next_state(PENDING, "enqueue_now"), READY)
        self.assertEqual(next_state(DELAYED, "due"), READY)
        self.assertEqual(next_state(READY, "claim"), INFLIGHT)
        self.assertEqual(next_state(INFLIGHT, "ack"), ACKED)
        self.assertEqual(next_state(INFLIGHT, "nack_retry"), DELAYED)
        self.assertEqual(next_state(INFLIGHT, "nack_dead"), DEAD)

    def test_visibility_timeout_returns_to_ready(self):
        self.assertEqual(
            next_state(INFLIGHT, "visibility_timeout"), READY)

    def test_illegal_transitions_rejected(self):
        for reason in ("ack", "nack_retry", "claim", "visibility_timeout"):
            with self.assertRaises(IllegalTransitionError):
                next_state(ACKED, reason)
        with self.assertRaises(IllegalTransitionError):
            next_state(DEAD, "claim")
        with self.assertRaises(IllegalTransitionError):
            next_state(READY, "ack")

    def test_permission_table_covers_six_states(self):
        from delayqueue.models import TRANSITIONS
        touched = {source for source, _ in TRANSITIONS} | set(
            TRANSITIONS.values())
        for state in (PENDING, DELAYED, READY, INFLIGHT, ACKED, DEAD):
            self.assertIn(state, touched)


if __name__ == "__main__":
    unittest.main()
