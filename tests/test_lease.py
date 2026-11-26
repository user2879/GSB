import os
import unittest

from leasebatch.errors import LeaseError
from leasebatch.lease import LeaseManager
from leasebatch.spec import load_spec_dir


class FakeClock:
    def __init__(self, start=1000):
        self.value = start

    def now_ms(self):
        return self.value

    def advance(self, ms):
        self.value += ms


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec_dir(
            os.path.join(os.path.dirname(__file__), "..", "samples")
        )
        self.clock = FakeClock()
        self.manager = LeaseManager(self.spec, self.clock)

    def test_grant_respects_global_and_worker_caps(self):
        first = self.manager.grant("w1", "b1")
        self.manager.grant("w2", "b2")
        self.assertEqual(self.manager.active_count(), 2)
        with self.assertRaises(LeaseError) as caught:
            self.manager.grant("w1", "b3")
        self.assertEqual(caught.exception.code, "E_LEASE")
        self.manager.cancel(first.lease_id)
        third = self.manager.grant("w1", "b3")
        self.assertTrue(third.active)
        self.assertEqual(self.manager.active_count(), 2)

    def test_expiry_uses_monotonic_deadline(self):
        lease = self.manager.grant("w1", "b1")
        self.clock.advance(499)
        self.assertEqual(self.manager.sweep_expired(), [])
        self.clock.advance(2)
        self.assertEqual(self.manager.sweep_expired(), [lease.lease_id])
        self.assertEqual(self.manager.active_count(), 0)

    def test_optimistic_renewal_version_conflict_rejects(self):
        lease = self.manager.grant("w1", "b1")
        renewed = self.manager.renew(lease.lease_id, 1)
        self.assertEqual(renewed.version, 2)
        before = renewed.expiry_ms
        self.clock.advance(5)
        with self.assertRaises(LeaseError):
            self.manager.renew(lease.lease_id, 1)
        # Stale renewal must not move the deadline.
        self.assertEqual(self.manager.get(lease.lease_id).expiry_ms, before)

    def test_expiry_and_cancel_have_single_owner(self):
        lease = self.manager.grant("w1", "b1")
        self.clock.advance(1000)
        self.manager.sweep_expired()
        # The slot was already reclaimed by expiry; cancel cannot own it.
        self.assertFalse(self.manager.cancel(lease.lease_id))
        snapshot = self.manager.get(lease.lease_id).snapshot()
        releases = [h for h in snapshot["history"] if h["event"] == "release"]
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0]["cause"], "expiry")
        # Reclaimed slot is reusable exactly once.
        new_lease = self.manager.grant("w1", "b2")
        self.assertEqual(new_lease.slot_index, lease.slot_index)


if __name__ == "__main__":
    unittest.main()
