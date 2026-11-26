import time
import unittest

from shardcast.errors import LeaseError
from shardcast.leases import LeaseManager


class LeaseManagerTests(unittest.TestCase):
    def test_grant_release_and_unique_owner(self) -> None:
        manager = LeaseManager(ttl_ms=1000)
        lease = manager.grant("l1", "s1", "n1", 1)
        self.assertEqual(lease.lease_version, 1)
        self.assertIsNotNone(manager.owner_of("s1"))
        with self.assertRaises(LeaseError):
            manager.grant("l2", "s1", "n2", 1)
        manager.release("l1")
        self.assertIsNone(manager.owner_of("s1"))

    def test_real_monotonic_expiry(self) -> None:
        manager = LeaseManager(ttl_ms=30)
        lease = manager.grant("l1", "s1", "n1", 1)
        self.assertFalse(manager.is_expired(lease.lease_id))
        time.sleep(0.05)
        self.assertTrue(manager.is_expired(lease.lease_id))
        expired = manager.expire_due()
        self.assertEqual([item.lease_id for item in expired], ["l1"])
        self.assertIsNone(manager.owner_of("s1"))

    def test_optimistic_renewal_requires_successor_version(self) -> None:
        manager = LeaseManager(ttl_ms=1000)
        lease = manager.grant("l1", "s1", "n1", 1)
        bad_receipt = {
            "accepted": False,
            "lease_version": lease.lease_version,
            "expire_ms": lease.expire_ms,
        }
        with self.assertRaises(LeaseError):
            manager.apply_renewal_receipt("l1", bad_receipt)
        current = manager.get("l1")
        self.assertEqual(current.lease_version, 1)
        good_receipt = {
            "accepted": True,
            "lease_version": 2,
            "expire_ms": current.expire_ms + 500,
        }
        renewed = manager.apply_renewal_receipt("l1", good_receipt)
        self.assertEqual(renewed.lease_version, 2)
        self.assertEqual(renewed.renewals, 1)

    def test_capacity_revocation_releases_only_older_leases(self) -> None:
        manager = LeaseManager(ttl_ms=1000)
        old = manager.grant("l1", "s1", "n1", 1)
        manager.release("l1")
        new = manager.grant("l2", "s2", "n1", 2)
        revoked = manager.revoke_before_version(2)
        self.assertEqual([lease.lease_id for lease in revoked], [])
        revoked = manager.revoke_before_version(3)
        self.assertEqual([lease.lease_id for lease in revoked], [new.lease_id])
        self.assertIsNone(manager.owner_of("s2"))
        self.assertEqual(manager.get("l2").state, "revoked")

    def test_every_decision_has_unique_adjudication(self) -> None:
        manager = LeaseManager(ttl_ms=1000)
        manager.grant("l1", "s1", "n1", 1)
        manager.release("l1", reason="done")
        manager.grant("l2", "s2", "n1", 1)
        ids = [decision.adjudication for decision in manager.decisions]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
