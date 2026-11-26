from __future__ import annotations

import time
import unittest

from gsb.errors import LeaseError
from gsb.lease import LeaseManager


class LeaseTests(unittest.TestCase):
    def test_wrong_version_does_not_extend_expiry(self) -> None:
        manager = LeaseManager(ttl_ms=10_000)
        lease = manager.grant(
            shard_id="s1",
            node_id="n1",
            capacity_version=1,
            resource_units=3,
            now_ms=1_000,
        )
        original_expiry = lease.expires_monotonic_ms
        with self.assertRaises(LeaseError) as ctx:
            manager.renew_locally(lease.lease_id, 999, now_ms=1_100, ttl_ms=10_000)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(manager.get(lease.lease_id).expires_monotonic_ms, original_expiry)
        self.assertEqual(manager.get(lease.lease_id).lease_version, 1)

    def test_optimistic_renewal_increments_version(self) -> None:
        manager = LeaseManager(ttl_ms=100)
        lease = manager.grant(
            shard_id="s1",
            node_id="n1",
            capacity_version=1,
            resource_units=1,
            now_ms=0,
        )
        renewed = manager.renew_locally(
            lease.lease_id,
            1,
            now_ms=10,
            ttl_ms=500,
            new_version=2,
        )
        self.assertEqual(renewed.lease_version, 2)
        self.assertEqual(renewed.expires_monotonic_ms, 510)

    def test_real_monotonic_expiry(self) -> None:
        manager = LeaseManager(ttl_ms=40)
        lease = manager.grant(
            shard_id="slow",
            node_id="n1",
            capacity_version=1,
            resource_units=2,
        )
        self.assertEqual(manager.expire_due(), [])
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            expired = manager.expire_due()
            if expired:
                break
            time.sleep(0.01)
        else:
            self.fail("lease did not genuinely expire")
        self.assertEqual(expired[0].lease_id, lease.lease_id)
        self.assertFalse(manager.get(lease.lease_id).active)

    def test_release_once_prevents_double_capacity_credit(self) -> None:
        manager = LeaseManager(ttl_ms=1000)
        lease = manager.grant(
            shard_id="s1",
            node_id="n1",
            capacity_version=2,
            resource_units=7,
        )
        first = manager.release_once(lease.lease_id, "expired", status="expired")
        second = manager.release_once(lease.lease_id, "revoked", status="revoked")
        self.assertIsNotNone(first)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
