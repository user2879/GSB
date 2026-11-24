import time
import unittest

from shard_ledger.errors import LeaseError
from shard_ledger.lease import LeaseManager


class LeaseManagerTest(unittest.TestCase):
    def test_grant_tracks_node_count_and_global_units(self):
        manager = LeaseManager(ttl_ms=500)
        manager.grant("S1", "N1", 3)
        second = manager.grant("S2", "N1", 4)
        self.assertEqual(manager.active_count("N1"), 2)
        self.assertEqual(manager.global_units(), 7)
        manager.release(second.lease_id)
        self.assertEqual(manager.active_count("N1"), 1)
        self.assertEqual(manager.global_units(), 3)

    def test_lease_limit_and_global_budget_enforced(self):
        manager = LeaseManager(ttl_ms=500)
        self.assertTrue(manager.can_grant("N1", 5, 2, 10))
        manager.grant("S1", "N1", 6)
        self.assertFalse(manager.can_grant("N1", 5, 2, 10))
        self.assertFalse(manager.can_grant("N2", 5, 2, 10))
        self.assertTrue(manager.can_grant("N2", 4, 2, 10))

    def test_optimistic_renew_version_check(self):
        manager = LeaseManager(ttl_ms=500)
        lease = manager.grant("S1", "N1", 1)
        renewed = manager.renew(lease.lease_id, 1)
        self.assertEqual(renewed.version, 2)
        self.assertGreaterEqual(renewed.expires_ms, lease.expires_ms)
        with self.assertRaises(LeaseError) as ctx:
            manager.renew(lease.lease_id, 1)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        current_expiry = renewed.expires_ms
        self.assertEqual(manager._leases[lease.lease_id].expires_ms, current_expiry)

    def test_real_expiry_releases_resources(self):
        manager = LeaseManager(ttl_ms=40)
        lease = manager.grant("S1", "N1", 3)
        time.sleep(0.08)
        expired = manager.sweep_expired()
        self.assertEqual([item.lease_id for item in expired], [lease.lease_id])
        self.assertEqual(manager.active_count("N1"), 0)
        self.assertEqual(manager.global_units(), 0)
        with self.assertRaises(LeaseError):
            manager.renew(lease.lease_id, lease.version)

    def test_expiry_callback_fires_once(self):
        events = []
        manager = LeaseManager(ttl_ms=30, on_expire=lambda lease: events.append(lease.lease_id))
        lease = manager.grant("S1", "N1", 1)
        time.sleep(0.07)
        manager.sweep_expired()
        manager.sweep_expired()
        self.assertEqual(events, [lease.lease_id])


if __name__ == "__main__":
    unittest.main()
