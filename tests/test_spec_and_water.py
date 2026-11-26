import copy
import unittest

from shardmesh.errors import ShardMeshError
from shardmesh.generator import demo_spec_dict
from shardmesh.spec import parse_spec
from shardmesh.water import build_attribution, budget_boundary


class SpecValidationTests(unittest.TestCase):
    def setUp(self):
        self.raw = demo_spec_dict(ports=[45111, 45112, 45113])

    def test_valid_demo_spec(self):
        spec = parse_spec(self.raw)
        self.assertEqual(len(spec.nodes), 3)
        self.assertEqual(len(spec.shards), 8)
        self.assertIn("s01", spec.topology)

    def test_optional_host_and_node_reference_have_defaults(self):
        raw = {
            "global_budget": 2,
            "scale": 10,
            "nodes": [
                {"id": "n1", "port": 45211, "concurrency": 1, "lease_limit": 1, "seed": 1, "capacity_version": 1}
            ],
            "shards": [
                {"id": "s1", "deps": [], "rows": 4, "units": 1, "priority": 0}
            ],
            "events": [{"seq": 1, "type": "arrive", "shard_id": "s1"}],
        }
        spec = parse_spec(raw)
        self.assertEqual(spec.nodes[0].host, "127.0.0.1")
        self.assertEqual(spec.shards[0].node_id, "n1")

    def test_duplicate_node_id_reports_first_entry(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][1]["id"] = raw["nodes"][0]["id"]
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.category, "node")
        self.assertEqual(ctx.exception.index, 1)
        self.assertEqual(ctx.exception.field_name, "id")

    def test_host_must_be_literal_loopback(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][0]["host"] = "0.0.0.0"
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field_name, "host")
        self.assertEqual(ctx.exception.index, 0)

    def test_duplicate_port(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][1]["port"] = raw["nodes"][0]["port"]
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field_name, "port")

    def test_negative_quota_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][0]["concurrency"] = -1
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field_name, "concurrency")

    def test_missing_dependency_and_cycle(self):
        raw = copy.deepcopy(self.raw)
        raw["shards"][2]["deps"] = ["missing"]
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field_name, "deps")

        cyclic = copy.deepcopy(self.raw)
        by_id = {shard["id"]: shard for shard in cyclic["shards"]}
        by_id["s01"]["deps"] = ["s02"]
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(cyclic)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.category, "dependency_graph")

    def test_supported_scale_and_event_sequence(self):
        raw = copy.deepcopy(self.raw)
        raw["scale"] = 7
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field_name, "scale")

        bad_events = copy.deepcopy(self.raw)
        bad_events["events"][2]["seq"] = 1
        with self.assertRaises(ShardMeshError) as ctx:
            parse_spec(bad_events)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.category, "event")
        self.assertEqual(ctx.exception.seq, 1)
        self.assertEqual(ctx.exception.field_name, "seq")


class WaterTests(unittest.TestCase):
    def test_attribution_and_budget_threshold_use_integers(self):
        raw = demo_spec_dict()
        spec = parse_spec(raw)
        attribution = build_attribution(
            shard_id="s07",
            deps=spec.deps_map,
            scale=1000,
            active_count=2,
            global_budget=7,
            node_quota=1,
            node_lease_limit=2,
        )
        self.assertGreaterEqual(attribution["synthetic_bound_q"], 3)
        self.assertIn(
            attribution["dominant_cause"],
            {
                "single_value_quantization_q",
                "dependency_propagation_q",
                "aggregation_accumulation_q",
            },
        )
        boundary = budget_boundary(
            used_units=4,
            candidate_units=3,
            budget=7,
            scale=1000,
            bound_q=attribution["synthetic_bound_q"],
        )
        self.assertEqual(boundary["slack_units"], 0)
        self.assertTrue(boundary["ambiguous"])
        self.assertTrue(boundary["hard_admittable"])


if __name__ == "__main__":
    unittest.main()
