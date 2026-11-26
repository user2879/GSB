import unittest

from gsb.clock import now_ms
from gsb.errors import LeaseError
from gsb.lease import LeaseRegistry


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.registry = LeaseRegistry()
        self.lease = self.registry.grant("sub-alpha", "batch-1", ttl_ms=80)

    def test_grant_fields(self):
        self.assertEqual(self.lease.version, 1)
        self.assertEqual(self.lease.expires_at_ms - self.lease.granted_at_ms, 80)

    def test_optimistic_renew_advances_version(self):
        renewed = self.registry.renew(self.lease.lease_id, 1, 80)
        self.assertEqual(renewed.version, 2)
        self.assertGreaterEqual(renewed.expires_at_ms, self.lease.expires_at_ms)

    def test_stale_version_rejected_and_not_extended(self):
        before = self.registry.for_batch("batch-1").expires_at_ms
        with self.assertRaises(LeaseError) as ctx:
            self.registry.renew(self.lease.lease_id, 99, 80)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        after = self.registry.for_batch("batch-1")
        self.assertEqual(after.version, 1)
        self.assertEqual(after.expires_at_ms, before)

    def test_real_monotonic_expiry(self):
        lease = self.registry.grant("sub-beta", "batch-2", ttl_ms=20)
        self.assertFalse(self.registry.is_expired(lease, now_ms()))
        deadline = now_ms() + 100
        while now_ms() < deadline:
            if self.registry.is_expired(lease):
                break
        self.assertTrue(self.registry.is_expired(lease))
        self.assertIn("batch-2", self.registry.expired_batches())


if __name__ == "__main__":
    unittest.main()
