from __future__ import annotations

import time
import unittest

from quota_ledger.errors import BusinessError
from quota_ledger.leases import LeaseManager


class LeaseTests(unittest.TestCase):
    def test_unique_active_lease_per_shard(self) -> None:
        manager = LeaseManager()
        first = manager.issue(
            lease_id="l1", shard_id="s1", node_id="n1", config_version=1, generation=1, ttl_ms=1000
        )
        self.assertEqual(first.version, 1)
        with self.assertRaises(BusinessError) as ctx:
            manager.issue(
                lease_id="l2", shard_id="s1", node_id="n2", config_version=1, generation=1, ttl_ms=1000
            )
        self.assertEqual(ctx.exception.code, "E_LEASE")

    def test_real_expiry_transitions_exactly_once(self) -> None:
        expired = []
        manager = LeaseManager(on_expired=expired.append)
        lease = manager.issue(
            lease_id="l1", shard_id="s1", node_id="n1", config_version=1, generation=1, ttl_ms=25
        )
        time.sleep(0.05)
        first = manager.expire_due()
        second = manager.expire_due()
        self.assertEqual([item.lease_id for item in first], [lease.lease_id])
        self.assertEqual(second, [])
        self.assertEqual(manager.node_counts("n1"), 0)

    def test_wrong_renewal_version_does_not_extend(self) -> None:
        manager = LeaseManager()
        lease = manager.issue(
            lease_id="l1", shard_id="s1", node_id="n1", config_version=1, generation=1, ttl_ms=1000
        )
        before = lease.expires_ms
        with self.assertRaises(BusinessError) as ctx:
            manager.renew("l1", 99, ttl_ms=5000)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(lease.expires_ms, before)
        self.assertEqual(lease.version, 1)
        renewed = manager.renew("l1", 1, ttl_ms=5000)
        self.assertEqual(renewed.version, 2)
        self.assertGreaterEqual(renewed.expires_ms, before)


if __name__ == "__main__":
    unittest.main()
