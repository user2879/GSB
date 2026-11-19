from __future__ import annotations

import struct
import unittest

from story_workbench.engine import project_node_outgoing_effects
from story_workbench.timeline import NarrativeSession

try:
    from tests.helpers import boundary, normal
except ModuleNotFoundError:  # supports `unittest discover -s tests`
    from helpers import boundary, normal


class DeterministicEngineTests(unittest.TestCase):
    def test_normal_state_and_conflict_arbitration(self):
        result = NarrativeSession(normal(), "normal_path").run()
        self.assertEqual(result.ending, "alliance")
        self.assertEqual(result.status, "ended")
        self.assertEqual(result.final_state, (0.30000000000000004, 0.5, 0.1))
        first_after_edge = result.snapshots[1]
        self.assertEqual(first_after_edge.state, (0.2, 0.30000000000000004, 0.05))
        self.assertEqual(len(first_after_edge.arbitrations), 2)
        ids = {
            (item.overwritten_effect_id, item.winner_effect_id, item.target)
            for item in first_after_edge.arbitrations
        }
        self.assertIn(("f_intro_trust_shadow", "f_intro_trust", "trust"), ids)
        self.assertIn(("f_intro_rep_shadow", "f_intro_rep", "reputation"), ids)

    def test_effect_rhs_uses_pre_edge_state(self):
        result = NarrativeSession(normal(), "normal_path").run()
        chain = {item.component: item for item in result.snapshots[1].causal_chains}
        fear_contribution = chain["fear"].contributions[0]
        self.assertEqual(fear_contribution.delta, 0.05)
        self.assertEqual(chain["trust"].edge_ids(), ("e_intro",))
        self.assertEqual(chain["reputation"].edge_ids(), ("e_intro",))

    def test_replay_is_bit_identical(self):
        first = NarrativeSession(normal(), "normal_path").run()
        second = NarrativeSession(normal(), "normal_path").run()
        self.assertEqual(first.edge_history, second.edge_history)
        self.assertEqual(first.edge_sequence_hash, second.edge_sequence_hash)
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.replay_hash, second.replay_hash)
        self.assertEqual(
            b"".join(struct.pack(">d", value) for value in first.final_state),
            b"".join(struct.pack(">d", value) for value in second.final_state),
        )
        self.assertEqual(
            [snapshot.content_hash for snapshot in first.snapshots],
            [snapshot.content_hash for snapshot in second.snapshots],
        )

    def test_boundary_is_a_separate_status_with_versioned_manifest(self):
        result = NarrativeSession(boundary(), "boundary_path").run()
        gate_snapshot = next(snapshot for snapshot in result.snapshots if snapshot.node_id == "b_gate")
        boundary_branch = next(branch for branch in gate_snapshot.branches if branch.edge_id == "e_b_boundary")
        clear_branch = next(branch for branch in gate_snapshot.branches if branch.edge_id == "e_b_clear")
        self.assertEqual(boundary_branch.status, "boundary")
        self.assertEqual(clear_branch.status, "reachable")
        judgment = boundary_branch.conditions[0]
        self.assertTrue(judgment.boundary)
        self.assertEqual(judgment.status, "boundary")
        self.assertEqual(judgment.value, 0.30000000000000004)
        self.assertEqual(judgment.threshold, 0.3)
        self.assertEqual(judgment.signed_margin, 5.551115123125783e-17)
        manifest = gate_snapshot.as_dict()["boundary_manifest"]
        self.assertEqual(manifest["state_version"], gate_snapshot.state_version)
        self.assertEqual(manifest["content_hash"], gate_snapshot.content_hash)
        self.assertEqual(manifest["version_stamp"], gate_snapshot.version_stamp)
        self.assertEqual(manifest["judgments"][0]["condition_id"], "c_b_boundary")

    def test_component_order_controls_chain_order(self):
        result = NarrativeSession(normal(), "normal_path").run()
        self.assertEqual([chain.component for chain in result.snapshots[-1].causal_chains], ["trust", "reputation", "fear"])

    def test_node_outgoing_effect_group_uses_edge_then_effect_last_writer(self):
        spec = normal()
        snapshot = NarrativeSession(spec, "normal_path").run().snapshots[1]
        projection = project_node_outgoing_effects(spec, "s_choice", snapshot.state)
        self.assertEqual(projection["edge_order"], ["e_ally", "e_wait", "e_fail"])
        self.assertEqual(projection["projected_state"]["trust"], 0.25)
        self.assertEqual(projection["projected_state"]["reputation"], 0.30000000000000004)
        self.assertEqual(projection["projected_state"]["fear"], 0.25)
        winners = {
            (item["target"], item["winner_effect_id"])
            for item in projection["arbitrations"]
        }
        self.assertIn(("trust", "f_wait_trust"), winners)
        self.assertIn(("reputation", "f_fail_rep"), winners)
        self.assertIn(("fear", "f_fail_fear"), winners)


if __name__ == "__main__":
    unittest.main()
