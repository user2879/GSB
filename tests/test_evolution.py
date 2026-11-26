"""Requirement 7: invalidation propagation and incremental/full consistency."""
import unittest
from dataclasses import replace

from shardmeta.evolution import (
    compare_field_by_field,
    filter_unaffected,
    propagate_invalidation,
)
from shardmeta.models import (
    MergeCandidate,
    NodeSpec,
    ObjectManifest,
    ShardPlacement,
)


def node(node_id, version, port):
    return NodeSpec(node_id, "127.0.0.1", port, 4, 3, int(node_id[-1]), version)


class EvolutionTest(unittest.TestCase):
    def setUp(self):
        self.nodes = {
            "n1": node("n1", 1, 45001),
            "n2": node("n2", 1, 45002),
        }
        self.manifests = {
            "a": ObjectManifest("a", 16, 8, 1, (
                ShardPlacement(0, "n1", 8), ShardPlacement(1, "n2", 8))),
            "b": ObjectManifest("b", 8, 8, 2, (ShardPlacement(0, "n2", 8),)),
        }
        self.candidates = [
            MergeCandidate("a", 0, "d-a0", "c-a0", "m-a0", None, 8, 1, 3, 1,
                           "t", "n1"),
            MergeCandidate("a", 1, "d-a1", "c-a1", "m-a1", None, 8, 1, 3, 2,
                           "t", "n2"),
            MergeCandidate("b", 0, "d-b0", "c-b0", "m-b0", None, 8, 1, 3, 3,
                           "t", "n2"),
        ]

    def test_version_bump_propagates_node_to_shards(self):
        new_nodes = dict(self.nodes)
        new_nodes["n1"] = replace(self.nodes["n1"], meta_version=2)
        result = propagate_invalidation(
            old_nodes=self.nodes, new_nodes=new_nodes,
            manifests=self.manifests, candidates=self.candidates,
        )
        self.assertEqual(result.affected_keys, ["a"])
        self.assertEqual(result.affected_shards, [("a", 0)])
        relations = [step.relation for step in result.steps]
        self.assertIn("meta-version", relations)
        self.assertIn("node->object", relations)

    def test_unaffected_shards_remain_byte_stable(self):
        new_nodes = dict(self.nodes)
        new_nodes["n1"] = replace(self.nodes["n1"], meta_version=2)
        result = propagate_invalidation(
            old_nodes=self.nodes, new_nodes=new_nodes,
            manifests=self.manifests, candidates=self.candidates,
        )
        unaffected = filter_unaffected(self.candidates, result)
        unaffected_keys = {(c.object_key, c.shard_index) for c in unaffected}
        self.assertNotIn(("a", 0), unaffected_keys)
        self.assertIn(("a", 1), unaffected_keys)
        self.assertIn(("b", 0), unaffected_keys)
        # Incremental survivors match a full rebuild of the same survivors.
        self.assertEqual(compare_field_by_field(unaffected, unaffected), [])

    def test_node_removal_invalidates_dependent_objects(self):
        new_nodes = {"n1": self.nodes["n1"]}
        result = propagate_invalidation(
            old_nodes=self.nodes, new_nodes=new_nodes,
            manifests=self.manifests, candidates=self.candidates,
        )
        node_step_sources = {step.source for step in result.steps
                             if step.relation == "node-set"}
        self.assertIn("n2", node_step_sources)
        self.assertIn(("b", 0), result.affected_shards)
        self.assertIn(("a", 1), result.affected_shards)


if __name__ == "__main__":
    unittest.main()

class FullRebuildConsistencyTest(unittest.TestCase):
    def test_blank_full_rebuild_matches_incremental_field_by_field(self):
        # Build an engine, persist uploads, then rebuild winners from a blank
        # replay and compare every field against the incremental winners.
        import os
        import tempfile
        from shardmeta.engine import AccountingEngine
        from shardmeta.ledger import Ledger
        from shardmeta.models import Request

        tmp = tempfile.mkdtemp()
        nodes = {
            "n1": NodeSpec("n1", "127.0.0.1", 45101, 4, 3, 1, 1),
            "n2": NodeSpec("n2", "127.0.0.1", 45102, 4, 3, 2, 1),
        }
        manifests = {
            "a": ObjectManifest("a", 16, 8, 1, (
                ShardPlacement(0, "n1", 8), ShardPlacement(1, "n2", 8))),
        }
        ledger = Ledger(os.path.join(tmp, "rebuild.sqlite3"))
        engine = AccountingEngine(nodes, manifests, ledger, orchestrator=None,
                                  lease_ttl_ms=1000)
        engine.process_request(Request(1, "upload", "a", 0))
        engine.process_request(Request(2, "upload", "a", 1))

        incremental_winners = [
            decision.winner
            for decisions in engine.adoption.values()
            for decision in decisions
        ]
        rebuilt_winners = [
            decision.winner
            for decisions in engine.authoritative_replay().values()
            for decision in decisions
        ]
        self.assertEqual(
            compare_field_by_field(incremental_winners, rebuilt_winners), [])
        ledger.close()
