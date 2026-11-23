import unittest

from arcore.clock import FakeClock
from arcore.errors import LeaseError
from arcore.lease import BatchRange, LeaseManager


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(1000)
        self.manager = LeaseManager(
            global_capacity=2, service_capacity=1,
            ttl_ms=100, clock=self.clock)
        self.range_a = BatchRange("svc-a", 0, 5)
        self.range_b = BatchRange("svc-b", 0, 5)

    def test_grant_includes_full_metadata(self):
        lease = self.manager.grant(self.range_a, "holder-1")
        self.assertEqual(lease.version, 1)
        self.assertEqual(lease.granted_ms, 1000)
        self.assertEqual(lease.expires_ms, 1100)
        self.assertEqual(lease.service_id, "svc-a")

    def test_global_capacity_blocks_grant(self):
        self.manager.grant(self.range_a, "h1")
        self.manager.grant(self.range_b, "h2")
        with self.assertRaises(LeaseError) as ctx:
            self.manager.grant(BatchRange("svc-c", 0, 1), "h3")
        self.assertEqual(ctx.exception.code, "E_LEASE")

    def test_service_capacity_blocks_grant(self):
        self.manager.grant(self.range_a, "h1")
        with self.assertRaises(LeaseError):
            self.manager.grant(BatchRange("svc-a", 5, 9), "h2")

    def test_renew_requires_holder_and_version(self):
        lease = self.manager.grant(self.range_a, "holder-1")
        with self.assertRaises(LeaseError):
            self.manager.renew(lease.lease_id, "intruder", 1)
        with self.assertRaises(LeaseError):
            self.manager.renew(lease.lease_id, "holder-1", 99)
        renewed = self.manager.renew(
            lease.lease_id, "holder-1", 1, extension_ms=200)
        self.assertEqual(renewed.version, 2)
        self.assertEqual(renewed.expires_ms, 1200)

    def test_expiry_uses_real_elapsed_monotonic_time_and_reaps_quota(self):
        lease = self.manager.grant(self.range_a, "holder-1")
        self.clock.advance(100)
        self.assertTrue(lease.is_expired(self.clock.now_ms()))
        reaped = self.manager.reap_expired()
        self.assertEqual(len(reaped), 1)
        self.assertEqual(reaped[0].reaped_ms, 1100)
        self.assertEqual(self.manager.active_count(), 0)
        with self.assertRaises(LeaseError):
            self.manager.renew(lease.lease_id, "holder-1", 1)
        # Quota was recycled: a new lease can be granted.
        self.manager.grant(self.range_a, "holder-2")

    def test_release_returns_quota(self):
        lease = self.manager.grant(self.range_a, "holder-1")
        self.manager.release(lease.lease_id)
        self.assertEqual(self.manager.active_count(), 0)
        self.manager.grant(self.range_a, "holder-2")


if __name__ == "__main__":
    unittest.main()
