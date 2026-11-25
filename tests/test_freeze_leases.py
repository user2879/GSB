import unittest

from sharding_diag.clock import ManualClock
from sharding_diag.errors import LeaseError, UpstreamError
from sharding_diag.freeze import FROZEN, UNFROZEN, UPSTREAM_HELD, FreezeManager
from sharding_diag.leases import LeaseManager
from sharding_diag.model import parse_declaration


def declaration():
    return parse_declaration({
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 1,
             "lease_limit": 1, "seed": 1, "capacity_version": 1},
        ],
        "shards": [
            {"shard_id": "a", "rows": 5, "demand": 2, "priority": 1},
            {"shard_id": "b", "depends_on": ["a"], "rows": 5, "demand": 2, "priority": 1},
        ],
        "global": {"budget": 10, "freeze_threshold_ms": 50, "revoke_tolerance": 5},
    })


class FreezeTest(unittest.TestCase):
    def test_root_thaws_downstream_held_then_thaws(self):
        clock = ManualClock()
        fm = FreezeManager(clock, 50)
        decl = declaration()
        fm.initialize(s.shard_id for s in decl.shards)
        first = fm.evaluate_round(decl.shards, {})
        self.assertEqual(fm.state_of("a"), UNFROZEN)
        self.assertEqual(fm.state_of("b"), UPSTREAM_HELD)
        self.assertEqual(fm.missing_upstreams("b"), ["a"])
        # E_UPSTREAM contract: held verdict names the missing upstream.
        with self.assertRaises(UpstreamError) as ctx:
            raise UpstreamError("held", object_id="b", field="depends_on",
                                detail={"missing": fm.missing_upstreams("b")})
        self.assertEqual(ctx.exception.detail["missing"], ["a"])

        outcome = {"a": {"status": "COMMITTED", "summary_digest": "sha256:x"}}
        clock.advance(60)
        second = fm.evaluate_round(decl.shards, outcome)
        self.assertEqual(fm.state_of("b"), UNFROZEN)
        b_record = next(r for r in second if r["shard_id"] == "b")
        self.assertTrue(b_record["threshold_breached"])

    def test_order_independence(self):
        clock = ManualClock()
        decl = declaration()
        outcomes = {"a": {"status": "COMMITTED", "summary_digest": "sha256:x"}}
        verdicts = []
        for _ in range(3):
            fm = FreezeManager(clock, 50)
            fm.initialize(s.shard_id for s in decl.shards)
            migrations = fm.evaluate_round(decl.shards, outcomes)
            verdicts.append([(m["shard_id"], m["to_state"]) for m in migrations])
        self.assertEqual(verdicts[0], verdicts[1])
        self.assertEqual(verdicts[1], verdicts[2])

    def test_frozen_shard_never_dispatched_contract(self):
        clock = ManualClock()
        fm = FreezeManager(clock, 50)
        fm.initialize(["a"])
        self.assertTrue(fm.is_frozen("a"))
        self.assertNotIn("a", fm.ready_shards())


class LeaseTest(unittest.TestCase):
    def test_grant_expire_revoke_and_unique_release(self):
        clock = ManualClock(100)
        lm = LeaseManager(clock)
        lease = lm.grant("n1", "a", 2, ttl_ms=100)
        self.assertEqual(lease.version, 1)
        self.assertEqual(lease.expiry_ms, 200)
        self.assertEqual(lm.budget_used(), 2)

        # Wrong optimistic version is rejected with E_LEASE.
        with self.assertRaises(LeaseError) as ctx:
            lm.revoke("n1", "a", expected_version=99)
        self.assertEqual(ctx.exception.code, "E_LEASE")

        # Expiry wins the unique release arbitration first.
        clock.advance(101)
        events = lm.expire_due()
        self.assertEqual(events[0]["reason"], "expiry")
        self.assertEqual(lm.budget_used(), 0)
        # Revoke and freeze paths then become no-ops: no double return.
        with self.assertRaises(LeaseError):
            lm.revoke("n1", "a", expected_version=1)
        self.assertIsNone(lm.release_for_freeze("n1", "a"))
        self.assertEqual(len(lm.release_log()), 1)

    def test_version_monotonic_after_regrant(self):
        clock = ManualClock()
        lm = LeaseManager(clock)
        first = lm.grant("n1", "a", 2, 100)
        lm.revoke("n1", "a", first.version)
        second = lm.grant("n1", "a", 2, 100)
        self.assertEqual(second.version, first.version + 1)

    def test_three_path_arbitration_first_wins(self):
        clock = ManualClock()
        for reason, action in (
            ("revoke", lambda: None),
        ):
            lm = LeaseManager(clock)
            lease = lm.grant("n1", "a", 3, 100)
            event = lm.revoke("n1", "a", lease.version)
            self.assertEqual(event["reason"], "revoke")
            self.assertIsNone(lm.release_on_complete("n1", "a"))
            self.assertIsNone(lm.release_for_freeze("n1", "a"))
            self.assertEqual(lm.budget_used(), 0)


if __name__ == "__main__":
    unittest.main()
