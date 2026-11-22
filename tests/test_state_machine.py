import unittest

from distlock.arbiter.state_machine import DomainArbiter
from distlock.clock import FakeClock
from distlock.models import Domain, ScaleFactor

SCALE = ScaleFactor(10_000)
PRIO = {"a": 10, "b": 20, "c": 5, "e": 15}


def make(capacity=2, lease=100, queue=3):
    return DomainArbiter(
        Domain("d", capacity, lease, 20, queue, 1), SCALE, FakeClock())


class StateMachineTests(unittest.TestCase):
    def test_grant_queue_stable_drain(self):
        arb = make()
        self.assertEqual(arb.handle(1, "a", "ACQUIRE", PRIO["a"]).verdict, "GRANTED")
        self.assertEqual(arb.handle(2, "b", "ACQUIRE", PRIO["b"]).verdict, "GRANTED")
        self.assertEqual(arb.handle(3, "c", "ACQUIRE", PRIO["c"]).verdict, "QUEUED")
        self.assertEqual(arb.handle(4, "e", "ACQUIRE", PRIO["e"]).verdict, "QUEUED")
        # Higher priority e must be ahead of c regardless of arrival.
        self.assertEqual([w.holder_id for w in arb._waiters], ["e", "c"])

        r = arb.handle(5, "a", "RELEASE", PRIO["a"])
        self.assertEqual(r.verdict, "RELEASED")
        self.assertEqual([w.holder_id for w in arb._waiters], ["c"])
        self.assertIn("e", arb._leases)
        # Drain transitions are surfaced for cross-seq ledger persistence.
        self.assertTrue(any(t.holder_id == "e" for t in r.drain_transitions))

    def test_queue_full_rejection(self):
        arb = make(capacity=1, queue=1)
        arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        self.assertEqual(arb.handle(2, "b", "ACQUIRE", PRIO["b"]).verdict, "QUEUED")
        self.assertEqual(
            arb.handle(3, "c", "ACQUIRE", PRIO["c"]).verdict,
            "REJECTED_QUEUE_FULL")

    def test_renewal_before_and_after_expiry(self):
        clock = FakeClock()
        arb = DomainArbiter(Domain("d", 1, 100, 20, 2, 1), SCALE, clock)
        arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        clock.advance(50)
        self.assertEqual(arb.handle(2, "a", "RENEW", PRIO["a"]).verdict, "RENEWED")
        clock.advance(100)  # reach the renewed deadline
        transitions = arb.reap(seq=3)
        self.assertTrue(any(t.state_to == "expired" for t in transitions))
        late = arb.handle(3, "a", "RENEW", PRIO["a"])
        self.assertEqual(late.verdict, "RENEW_NOT_HELD")
        self.assertEqual(late.state, "released")

    def test_same_instant_cannot_be_both_expiry_and_renewal(self):
        clock = FakeClock()
        arb = DomainArbiter(Domain("d", 1, 100, 20, 2, 1), SCALE, clock)
        arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        clock.advance(100)  # now == expiry
        # reap runs first inside handle; renewal cannot win at the same instant.
        verdict = arb.handle(2, "a", "RENEW", PRIO["a"]).verdict
        self.assertEqual(verdict, "RENEW_NOT_HELD")

    def test_duplicate_redelivery_is_idempotent(self):
        arb = make()
        first = arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        second = arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        self.assertEqual(second.verdict, "DUPLICATE")
        self.assertEqual(second.remaining_ms, first.remaining_ms)

    def test_fixed_point_watermark_floor(self):
        arb = make(capacity=3)
        r = arb.handle(1, "a", "ACQUIRE", PRIO["a"])
        # 1/3 * 10000 floor = 3333, error bound strictly < 1 fixed unit.
        self.assertEqual(r.ratio_fixed, 3333)
        self.assertEqual(r.error_bound_fixed, 1)
        self.assertEqual(r.free_capacity, 2)


if __name__ == "__main__":
    unittest.main()
