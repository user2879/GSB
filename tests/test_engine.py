import unittest

from lease_diag.clock import VirtualClock
from lease_diag.engine import Engine
from lease_diag.errors import E_LEASE
from lease_diag.model import (
    EXPIRED, GRANTED, RENEWED, REVOKED, ROLLEDBACK, Pool, Request,
    EV_DENY, EV_EXPIRE, EV_GRANT, EV_RENEW, EV_REVOKE, EV_ROLLBACK,
)


def pool(pid="P1", capacity=2, lease_ms=100, priority=0, quota=2):
    return Pool(pid, capacity, lease_ms, 30, priority, quota)


def req(rid, key, pid="P1", rev=1, prio=1, arrive=0):
    return Request(rid, key, pid, rev, prio, arrive)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.clock = VirtualClock()
        self.engine = Engine({"P1": pool()}, {})

    def test_grant_then_renew_increments_count(self):
        decision = self.engine.submit(req("r1", "k1"), self.clock.now_ms())
        self.assertTrue(decision.accepted)
        lease = self.engine.leases[decision.lease_id]
        self.assertEqual(lease.state, GRANTED)
        self.clock.advance(20)
        decision = self.engine.submit(req("r2", "k1", rev=2),
                                      self.clock.now_ms())
        self.assertEqual(decision.event_type, EV_RENEW)
        self.assertEqual(lease.state, RENEWED)
        self.assertEqual(lease.renew_count, 1)

    def test_renewal_after_expiry_denied_and_lease_expires(self):
        self.engine.submit(req("r1", "k1"), 0)
        self.clock.advance(100)
        events = self.engine.tick(self.clock.now_ms())
        self.assertEqual(events[0].event_type, EV_EXPIRE)
        decision = self.engine.submit(req("r2", "k1"), self.clock.now_ms())
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, E_LEASE)
        # Expired lease keeps its history; a fresh grant creates a new lease.
        decision2 = self.engine.submit(req("r3", "k1"),
                                       self.clock.now_ms() + 1)
        self.assertTrue(decision2.accepted)
        self.assertNotEqual(decision.lease_id, decision2.lease_id)
        self.assertEqual(self.engine.leases[decision.lease_id].state, EXPIRED)

    def test_only_current_holder_may_renew(self):
        self.engine.submit(req("r1", "k1", prio=5), 0)
        decision = self.engine.submit(req("r2", "k1", prio=6, rev=2), 10)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, E_LEASE)

    def test_single_owner_per_pool_at_any_time(self):
        engine = Engine({"P1": pool(capacity=1)}, {})
        engine.submit(req("r1", "k1", "P1"), 0)
        decision = engine.submit(req("r2", "k2", "P1"), 5)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, E_LEASE)
        self.assertIn("no lower-priority", decision.detail["reason"])


class PreemptionTests(unittest.TestCase):
    def test_high_priority_preempts_low_priority_at_renewal_boundary(self):
        engine = Engine({"P1": pool(capacity=1)}, {})
        engine.submit(req("r1", "low", prio=1), 0)
        engine.submit(req("r2", "low", prio=1, rev=2), 20)
        victim = next(iter(engine.leases.values()))
        self.assertEqual(victim.revoke_boundary_ms, 20)
        decision = engine.submit(req("r3", "high", prio=9), 40)
        self.assertTrue(decision.accepted)
        self.assertEqual(victim.state, REVOKED)
        revoke_event = next(event for event in engine.events
                            if event.event_type == EV_REVOKE)
        # Revocation is recorded at the safe boundary, renewals are retained.
        self.assertEqual(revoke_event.at_ms, 20)
        self.assertEqual(revoke_event.detail["renew_count"], 1)
        self.assertEqual(victim.revoked_by_lease_id, decision.lease_id)
        self.assertEqual(len(engine.active_index), 1)

    def test_no_lower_priority_victim_produces_e_lease_with_conflict(self):
        engine = Engine({"P1": pool(capacity=1)}, {})
        engine.submit(req("r1", "low", prio=5), 0)
        decision = engine.submit(req("r2", "high", prio=5), 10)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, E_LEASE)
        self.assertTrue(decision.lease_id.startswith("L-low-P1"))

    def test_rollback_converges_all_active_leases_to_target_revision(self):
        engine = Engine(
            {"P1": pool(capacity=2), "P2": pool("P2", capacity=2)},
            {"low": {"from_rev": 3, "target_rev": 1}})
        # Same key holds in two pools; both unexpired leases must converge.
        engine.submit(req("r1", "low", "P1", rev=3, prio=1), 0)
        engine.submit(req("r2", "low", "P2", rev=3, prio=1), 5)
        engine.submit(req("rX", "filler", "P1", prio=1), 6)
        decision = engine.submit(req("r3", "high", "P1", prio=9), 20)
        self.assertTrue(decision.accepted)
        rolled = [lease for lease in engine.leases.values()
                  if lease.key == "low"]
        self.assertEqual(len(rolled), 2)
        self.assertTrue(all(lease.state == ROLLEDBACK for lease in rolled))
        self.assertTrue(all(lease.rolledback_rev == 1 for lease in rolled))
        kinds = [event.event_type for event in engine.events]
        self.assertIn(EV_ROLLBACK, kinds)
        self.assertIn(EV_REVOKE, kinds)
        # Capacity released: winner + the unrelated P2 lease remain active.
        self.assertEqual(len(engine.active_index), 2)

    def test_contention_converges_to_one_deterministic_winner(self):
        outcomes = set()
        for _ in range(5):
            engine = Engine({"P1": pool(capacity=1)}, {})
            engine.submit(req("r1", "low", prio=1), 0)
            decision = engine.submit(req("r2", "high", prio=9), 10)
            outcomes.add(decision.lease_id)
        self.assertEqual(len(outcomes), 1)


class PermissionTableTests(unittest.TestCase):
    def test_terminal_states_have_no_outgoing_transitions(self):
        from lease_diag.model import TRANSITIONS
        for state in (EXPIRED, REVOKED, ROLLEDBACK):
            self.assertEqual(TRANSITIONS[state], {})

    def test_grant_events_have_stable_lease_ids(self):
        engine = Engine({"P1": pool()}, {})
        first = engine.submit(req("r1", "k1"), 0)
        self.assertEqual(first.lease_id, "L-k1-P1-001")
        clock = VirtualClock()
        engine.tick(100)
        # First post-expiry request is an explicit conflict, then re-grant.
        denied = engine.submit(req("r2", "k1"), 100)
        self.assertFalse(denied.accepted)
        second = engine.submit(req("r3", "k1"), 101)
        self.assertEqual(second.lease_id, "L-k1-P1-002")


if __name__ == "__main__":
    unittest.main()
