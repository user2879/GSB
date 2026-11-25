import unittest

from rlcb.clock import FakeMonotonicClock
from rlcb.codes import E_LEASE, LeaseError
from rlcb.lease import (LeaseManager, LeaseRegistry, arbitrate_ownership,
                        REASON_EXPIRED, REASON_SUPERSEDED)


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeMonotonicClock(100)
        self.manager = LeaseManager(self.clock, ttl_ms=50)
        self.registry = LeaseRegistry("svcA", ttl_ms=50)

    def test_grant_records_version_and_event(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.assertEqual(lease.version, 1)
        self.assertEqual(lease.granted_ms, 100)
        self.assertEqual(lease.expires_ms, 150)
        self.registry.record_grant(lease.lease_id, 1)
        self.assertEqual(len(self.manager.events), 1)

    def test_real_clock_expiry_drives_reclamation(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.registry.record_grant(lease.lease_id, 1)
        self.clock.advance(49)
        self.assertFalse(self.manager.is_expired(lease))
        self.assertEqual(self.manager.reclaim_expired(), [])
        self.clock.advance(1)  # now == expiry
        self.assertTrue(self.manager.is_expired(lease))
        events = self.manager.reclaim_expired()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, REASON_EXPIRED)
        self.assertEqual(self.manager.active_count(), 0)

    def test_optimistic_renewal_stale_version_rejected(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.registry.record_grant(lease.lease_id, 1)
        # Server advanced to version 2; client still presents version 1.
        self.registry.bump(lease.lease_id)
        before = lease.expires_ms
        with self.assertRaises(LeaseError) as ctx:
            self.manager.renew(lease.lease_id, 1, self.registry)
        self.assertEqual(ctx.exception.code, E_LEASE)
        # Deadline must not move on a rejected renewal.
        self.assertEqual(self.manager._leases[lease.lease_id].expires_ms,
                         before)

    def test_successful_renewal_extends_deadline(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.registry.record_grant(lease.lease_id, 1)
        self.clock.advance(10)
        renewed = self.manager.renew(lease.lease_id, 1, self.registry)
        self.assertEqual(renewed.version, 2)
        self.assertEqual(renewed.expires_ms, 110 + 50)

    def test_renewal_after_expiry_rejected(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.registry.record_grant(lease.lease_id, 1)
        self.clock.advance(60)
        with self.assertRaises(LeaseError) as ctx:
            self.manager.renew(lease.lease_id, 1, self.registry)
        self.assertEqual(ctx.exception.code, E_LEASE)

    def test_double_grant_same_slot_rejected(self):
        self.manager.grant(service_id="svcA", batch_id="B1", slot_index=0,
                           quota_version=1, lease_seq=1)
        with self.assertRaises(LeaseError):
            self.manager.grant(service_id="svcA", batch_id="B2",
                               slot_index=0, quota_version=1, lease_seq=2)

    def test_expiry_and_supersede_never_double_count_slot(self):
        self.manager.grant(service_id="svcA", batch_id="B1",
                           slot_index=0, quota_version=1, lease_seq=1)
        self.clock.advance(60)  # expired
        # When both apply, expiry wins (single reclaim event for the slot).
        events = self.manager.reclaim_superseded(new_quota_version=2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, REASON_EXPIRED)
        self.assertNotIn(("svcA", 0), self.manager.active)

    def test_supersede_for_fresh_lease(self):
        self.manager.grant(service_id="svcA", batch_id="B1", slot_index=0,
                           quota_version=1, lease_seq=1)
        events = self.manager.reclaim_superseded(new_quota_version=2)
        self.assertEqual(events[0].event_type, REASON_SUPERSEDED)

    def test_arbitration_replay_order_independent(self):
        lease = self.manager.grant(service_id="svcA", batch_id="B1",
                                   slot_index=0, quota_version=1, lease_seq=1)
        self.registry.record_grant(lease.lease_id, 1)
        self.clock.advance(5)
        self.manager.renew(lease.lease_id, 1, self.registry)
        self.manager.release(lease.lease_id)
        canonical = arbitrate_ownership(self.manager.events)
        # Re-feed events shuffled; canonical event_seq ordering must restore
        # the identical final ownership picture (slot released -> absent).
        shuffled = list(reversed(self.manager.events))
        self.assertEqual(canonical, arbitrate_ownership(shuffled))
        self.assertEqual(canonical, {})


if __name__ == "__main__":
    unittest.main()
