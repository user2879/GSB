import unittest

from sharding_diag.invalidation import diff_declarations, propagate
from sharding_diag.model import parse_declaration
from sharding_diag.revocation import (
    Candidate, DISTINGUISHABLE, LEASE_VERSION_CONFLICT,
    TOLERANCE_INDISTINGUISHABLE, UPSTREAM_UNDETERMINED,
    adjudicate_shard, revocation_statistics,
)


def cand(version, total, node="n1", shard="s"):
    return Candidate(shard, node, version, 1, total, f"digest-{node}-{version}")


class RevocationTest(unittest.TestCase):
    def test_tolerance_indistinguishable_uses_version_desc(self):
        incumbent, challenger = cand(1, 1000), cand(2, 1500)
        decision = adjudicate_shard("s", [incumbent, challenger], 1, False, 1000)
        self.assertEqual(decision["cause"], TOLERANCE_INDISTINGUISHABLE)
        self.assertTrue(decision["tolerance_affected"])
        self.assertEqual(decision["winner"]["version"], 2)

    def test_distinguishable_higher_fixed_total_wins(self):
        decision = adjudicate_shard("s", [cand(1, 1000), cand(2, 5000)], 1, False, 1000)
        self.assertEqual(decision["cause"], DISTINGUISHABLE)
        self.assertEqual(decision["winner"]["fixed_total"], 5000)

    def test_lease_version_conflict(self):
        decision = adjudicate_shard("s", [cand(2, 1000)], expected_version=1,
                                    is_frozen=False, tolerance=1000)
        self.assertEqual(decision["cause"], LEASE_VERSION_CONFLICT)

    def test_frozen_isolated_from_statistics(self):
        decision = adjudicate_shard("s", [cand(1, 1000)], 1, True, 1000)
        self.assertEqual(decision["cause"], UPSTREAM_UNDETERMINED)
        stats = revocation_statistics([decision])
        self.assertEqual(stats["frozen_excluded"], 1)
        self.assertEqual(stats[DISTINGUISHABLE], 0)

    def test_zero_tolerance_still_distinguishes_equal_totals(self):
        decision = adjudicate_shard("s", [cand(1, 1000), cand(2, 1000)], 1, False, 0)
        self.assertEqual(decision["cause"], TOLERANCE_INDISTINGUISHABLE)
        self.assertEqual(decision["diff_fp"], 0)


def declaration():
    return parse_declaration({
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 2,
             "lease_limit": 2, "seed": 1, "capacity_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 19002, "concurrency": 2,
             "lease_limit": 2, "seed": 2, "capacity_version": 1},
        ],
        "shards": [
            {"shard_id": "a", "rows": 5, "demand": 2, "priority": 1},
            {"shard_id": "b", "depends_on": ["a"], "rows": 5, "demand": 2, "priority": 1},
            {"shard_id": "c", "depends_on": ["b"], "rows": 5, "demand": 2, "priority": 1},
            {"shard_id": "d", "rows": 5, "demand": 2, "priority": 1},
        ],
        "global": {"budget": 20, "freeze_threshold_ms": 100, "revoke_tolerance": 5},
    })


class InvalidationTest(unittest.TestCase):
    def test_capacity_bump_invalidates_assigned_shards_and_descendants(self):
        decl = declaration()
        assignments = {"a": "n1", "b": "n2", "c": "n2", "d": "n1"}
        evolved = parse_declaration({
            "nodes": [
                {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 2,
                 "lease_limit": 2, "seed": 1, "capacity_version": 2},
                {"node_id": "n2", "host": "127.0.0.1", "port": 19002, "concurrency": 2,
                 "lease_limit": 2, "seed": 2, "capacity_version": 1},
            ],
            "shards": [
                {"shard_id": "a", "rows": 5, "demand": 2, "priority": 1},
                {"shard_id": "b", "depends_on": ["a"], "rows": 5, "demand": 2, "priority": 1},
                {"shard_id": "c", "depends_on": ["b"], "rows": 5, "demand": 2, "priority": 1},
                {"shard_id": "d", "rows": 5, "demand": 2, "priority": 1},
            ],
            "global": {"budget": 20, "freeze_threshold_ms": 100, "revoke_tolerance": 5},
        })
        changes = diff_declarations(decl, evolved)
        self.assertEqual(changes[0].kind, "capacity_version")
        plan = propagate(evolved, changes, assignments)
        # n1 capacity bump directly invalidates a and d (node references);
        # b and c then propagate transitively through the dependency graph.
        self.assertEqual(plan["invalidated"], ["a", "b", "c", "d"])
        self.assertEqual(plan["directly_invalidated"], ["a", "d"])
        self.assertEqual(plan["untouched"], [])
        hops = {(b["shard_id"], b["step"]) for b in plan["basis"]}
        self.assertIn(("a", "direct"), hops)
        self.assertIn(("d", "direct"), hops)
        self.assertIn(("b", "propagate"), hops)
        self.assertIn(("c", "propagate"), hops)

    def test_demand_change_only_invalidates_shard_and_descendants(self):
        decl = declaration()
        assignments = {"a": "n1", "b": "n2", "c": "n2", "d": "n1"}
        raw = {
            "nodes": [
                {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 2,
                 "lease_limit": 2, "seed": 1, "capacity_version": 1},
                {"node_id": "n2", "host": "127.0.0.1", "port": 19002, "concurrency": 2,
                 "lease_limit": 2, "seed": 2, "capacity_version": 1},
            ],
            "shards": [
                {"shard_id": "a", "rows": 5, "demand": 3, "priority": 1},
                {"shard_id": "b", "depends_on": ["a"], "rows": 5, "demand": 2, "priority": 1},
                {"shard_id": "c", "depends_on": ["b"], "rows": 5, "demand": 2, "priority": 1},
                {"shard_id": "d", "rows": 5, "demand": 2, "priority": 1},
            ],
            "global": {"budget": 20, "freeze_threshold_ms": 100, "revoke_tolerance": 5},
        }
        evolved = parse_declaration(raw)
        changes = diff_declarations(decl, evolved)
        self.assertEqual(changes[0].kind, "demand")
        plan = propagate(evolved, changes, assignments)
        self.assertEqual(plan["invalidated"], ["a", "b", "c"])
        self.assertEqual(plan["untouched"], ["d"])


if __name__ == "__main__":
    unittest.main()
