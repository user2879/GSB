"""Requirement 3: lease grant/renew/abort and real monotonic expiry."""
import unittest

from shardmeta.clock import ManualClock
from shardmeta.errors import LeaseError
from shardmeta.lease import LeaseManager


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.clock = ManualClock(start_ms=10_000)
        self.manager = LeaseManager("n1", capacity_slots=2, lease_limit=2,
                                    clock=self.clock)

    def test_grant_tracks_capacity(self):
        lease_a = self.manager.grant("k", 0, ttl_ms=100)
        self.assertEqual(lease_a.lease_version, 1)
        self.assertEqual(self.manager.state.active_count, 1)
        self.manager.grant("k", 1, ttl_ms=100)
        with self.assertRaises(LeaseError) as ctx:
            self.manager.grant("k", 2, ttl_ms=100)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(ctx.exception.field, "capacity_slots")

    def test_renew_optimistic_version_conflict_does_not_extend(self):
        lease = self.manager.grant("k", 0, ttl_ms=100)
        original_expiry = lease.expires_ms
        with self.assertRaises(LeaseError) as ctx:
            self.manager.renew(lease.lease_id, expected_version=99, ttl_ms=100)
        self.assertEqual(ctx.exception.field, "lease_version")
        unchanged = self.manager.get(lease.lease_id)
        self.assertEqual(unchanged.expires_ms, original_expiry)
        self.assertEqual(unchanged.lease_version, 1)

        self.clock.advance(40)
        renewed = self.manager.renew(lease.lease_id, expected_version=1, ttl_ms=100)
        self.assertEqual(renewed.lease_version, 2)
        self.assertEqual(renewed.renew_count, 1)
        self.assertGreater(renewed.expires_ms, original_expiry)

    def test_real_expiry_releases_slot_without_sleep_emulation(self):
        lease = self.manager.grant("k", 0, ttl_ms=100)
        self.clock.advance(99)
        self.assertTrue(lease.is_active(self.clock.now_ms()))
        self.clock.advance(1)
        expired = self.manager.sweep()
        self.assertEqual(len(expired), 1)
        self.assertEqual(self.manager.state.active_count, 0)
        # A new grant can now reuse the freed slot/budget.
        self.manager.grant("k", 0, ttl_ms=100)
        self.assertEqual(self.manager.state.active_count, 1)

    def test_abort_releases_slot(self):
        lease = self.manager.grant("k", 0, ttl_ms=100)
        aborted = self.manager.abort(lease.lease_id)
        self.assertEqual(aborted.state, "aborted")
        self.assertEqual(self.manager.state.active_count, 0)


if __name__ == "__main__":
    unittest.main()
