from __future__ import annotations

import dataclasses
import unittest

from shard_diag.diagnostics import build_replay_view
from shard_diag.errors import DivergeError
from shard_diag.invalidation import (
    ConclusionStore,
    assert_matches_full_rebuild,
    plan_change,
)
from shard_diag.spec import parse_spec
from shard_diag.topology import stable_topological_order
from shard_diag.water import (
    budget_slack_assessment,
    candidate_water,
    compare_candidate_water,
    global_budget_error_bound,
)

from tests.factories import sample_spec_dict


class WaterErrorTests(unittest.TestCase):
    def test_quantization_boundary_makes_allow_and_deny_indistinguishable(self) -> None:
        allowed = candidate_water(
            node_id="node-a",
            shard_id="allow",
            delay_ms=8,
            jitter_ms=0,
            revision_lag=2,
            max_revision_lag=2,
            delay_limit_ms=8,
            resource_units=1,
            lease_limit=4,
            scale_power=0,
        )
        denied = candidate_water(
            node_id="node-b",
            shard_id="deny",
            delay_ms=9,
            jitter_ms=0,
            revision_lag=1,
            max_revision_lag=2,
            delay_limit_ms=8,
            resource_units=1,
            lease_limit=4,
            scale_power=0,
        )
        self.assertEqual((allowed.water_q, denied.water_q), (2, 2))
        self.assertTrue(allowed.exact_delay_allowed)
        self.assertFalse(denied.exact_delay_allowed)
        comparison = compare_candidate_water(allowed, denied, 3, 4)
        self.assertTrue(comparison.indistinguishable)
        self.assertEqual(comparison.winner, "allow")
        self.assertEqual(comparison.stable_tie_winner, "allow")

    def test_error_segments_have_explicit_dominant_source(self) -> None:
        quant = candidate_water(
            node_id="n", shard_id="q", delay_ms=8, jitter_ms=0, revision_lag=0,
            max_revision_lag=1, delay_limit_ms=8, resource_units=1, lease_limit=4,
            scale_power=0,
        )
        jitter = candidate_water(
            node_id="n", shard_id="j", delay_ms=8, jitter_ms=12, revision_lag=0,
            max_revision_lag=1, delay_limit_ms=20, resource_units=10, lease_limit=2,
            scale_power=2,
        )
        lease = candidate_water(
            node_id="n", shard_id="l", delay_ms=8, jitter_ms=0, revision_lag=0,
            max_revision_lag=1, delay_limit_ms=20, resource_units=10, lease_limit=2,
            scale_power=2, lease_rearrangements=30,
        )
        self.assertEqual(quant.dominant_segment, "quantization_truncation")
        self.assertEqual(jitter.dominant_segment, "replay_observation_jitter")
        self.assertEqual(lease.dominant_segment, "lease_rearrangement")
        self.assertIn("0.60", quant.dominance_threshold)

    def test_global_budget_bound_and_slack_flag(self) -> None:
        bound = global_budget_error_bound(shard_count=6, scale_power=0)
        self.assertEqual(bound["hard_resource_quantization_bound_q"], 0)
        self.assertEqual(bound["aggregate_water_truncation_bound_q"], 1)
        self.assertEqual(bound["distributed_per_shard_sum_bound_q"], 6)
        slack = budget_slack_assessment(
            remaining_units=5,
            demand_units=4,
            scale_power=0,
            global_error_bound_q=bound["safe_upper_bound_q"],
        )
        self.assertEqual(slack["slack_units"], 1)
        self.assertTrue(slack["indistinguishable"])
        self.assertTrue(slack["hard_allowed"])


class InvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = parse_spec(sample_spec_dict())
        self.topo = stable_topological_order(self.spec.shards)
        self.store = ConclusionStore()
        for shard_id in self.topo:
            view = build_replay_view(self.spec, shard_id)
            record = {"view": view, "lease_owner": {"shard_id": shard_id, "epoch": 1}}
            self.store.put_current(shard_id, record)

    def test_capacity_change_propagates_node_refs_and_dependents(self) -> None:
        plan = plan_change(
            self.spec, {"type": "capacity_version", "node_id": "node-a", "new_version": 4}
        )
        self.assertEqual(
            plan.invalid_shards,
            ("s1-extract-a", "s3-join", "s5-boundary-allow", "s4-aggregate"),
        )
        self.assertTrue(all(step["basis"] for step in plan.propagation_steps))

        untouched = ["s2-extract-b", "s6-boundary-deny-quantized-equal"]
        before = {sid: self.store.current_bytes(sid) for sid in untouched}
        marked = self.store.mark_stale(plan, new_generation=4)
        self.assertEqual(len(marked), 4)
        self.assertNotIn("s1-extract-a", self.store.current_view())

        new_nodes = []
        for node in self.spec.nodes:
            if node.node_id == "node-a":
                node = dataclasses.replace(node, capacity_version=4)
            new_nodes.append(node)
        new_spec = dataclasses.replace(self.spec, nodes=tuple(new_nodes))

        for shard_id in plan.invalid_shards:
            view = build_replay_view(new_spec, shard_id)
            record = {"view": view, "lease_owner": {"shard_id": shard_id, "epoch": 2}}
            self.store.put_current(shard_id, record)

        for sid in untouched:
            self.assertEqual(self.store.current_bytes(sid), before[sid])

        views = {
            sid: self.store.current_record(sid)["view"] for sid in self.topo
        }
        assert_matches_full_rebuild(new_spec, views, order=self.topo)

    def test_resource_change_and_scale_change_sets(self) -> None:
        resource_plan = plan_change(
            self.spec, {"type": "resource_units", "shard_id": "s1-extract-a", "new_value": 9}
        )
        self.assertEqual(
            resource_plan.invalid_shards,
            ("s1-extract-a", "s3-join", "s4-aggregate"),
        )
        scale_plan = plan_change(self.spec, {"type": "scale_factor", "new_power": 2})
        self.assertEqual(tuple(scale_plan.invalid_shards), tuple(self.topo))
        self.assertIn("water", scale_plan.invalid_scopes)

        revision_plan = plan_change(
            self.spec, {"type": "revision_chain", "node_id": "node-b"}
        )
        self.assertEqual(
            revision_plan.invalid_shards,
            ("s2-extract-b", "s3-join", "s6-boundary-deny-quantized-equal", "s4-aggregate"),
        )
        global_revision = plan_change(self.spec, {"type": "revision_chain"})
        self.assertEqual(tuple(global_revision.invalid_shards), tuple(self.topo))

    def test_diverge_error_locates_first_field(self) -> None:
        views = {sid: build_replay_view(self.spec, sid) for sid in self.topo}
        first = self.topo[0]
        views[first]["summary"]["participating_rows"] += 1
        with self.assertRaises(DivergeError) as raised:
            assert_matches_full_rebuild(self.spec, views, order=self.topo)
        self.assertEqual(raised.exception.object, first)
        self.assertIn("participating_rows", raised.exception.field)


if __name__ == "__main__":
    unittest.main()
