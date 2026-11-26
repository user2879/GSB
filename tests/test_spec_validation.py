import copy
import unittest

from shard_diag.errors import SpecError
from shard_diag.spec import parse_events, parse_spec


def valid_spec_dict():
    return {
        "format_version": 1,
        "global_budget": 10,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 39501,
                "concurrency_quota": 2,
                "max_leases_per_node": 2,
                "seed": 7,
                "quota_version": 1,
            }
        ],
        "shards": [
            {"shard_id": "a", "dependencies": [], "sample_rows": 8, "resource_units": 2, "priority": 1},
            {"shard_id": "b", "dependencies": ["a"], "sample_rows": 8, "resource_units": 1, "priority": 2},
        ],
    }


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec(self):
        spec = parse_spec(valid_spec_dict())
        self.assertEqual([s.shard_id for s in spec.shards], ["a", "b"])
        self.assertEqual(spec.nodes[0].host, "127.0.0.1")

    def test_duplicate_node_id_reports_first_object(self):
        raw = valid_spec_dict()
        raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
        raw["nodes"][1]["port"] = 39502
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "nodes[1].node_id")
        self.assertEqual(ctx.exception.object_id, "n1")

    def test_host_literal_and_duplicate_port(self):
        raw = valid_spec_dict()
        raw["nodes"][0]["host"] = "0.0.0.0"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "nodes[0].host")
        raw = valid_spec_dict()
        raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
        raw["nodes"][1]["node_id"] = "n2"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "nodes[1].port")

    def test_bool_is_not_integer(self):
        raw = valid_spec_dict()
        raw["nodes"][0]["concurrency_quota"] = True
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertIn("concurrency_quota", ctx.exception.field)

    def test_missing_dependency(self):
        raw = valid_spec_dict()
        raw["shards"][1]["dependencies"] = ["missing"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.object_id, "b")
        self.assertIn("dependencies", ctx.exception.field)

    def test_cycle(self):
        raw = valid_spec_dict()
        raw["shards"][0]["dependencies"] = ["b"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertIn("cycle", str(ctx.exception))

    def test_event_seq_and_reference_validation(self):
        spec = parse_spec(valid_spec_dict())
        events = parse_events({"events": [
            {"seq": 1, "type": "dispatch", "shard_id": "a", "node_id": "n1"},
            {"seq": 2, "type": "priority", "shard_id": "a", "priority": 9},
        ]})
        self.assertEqual(len(events.events), 2)
        from shard_diag.spec import validate_event_references
        validate_event_references(events, spec)
        bad = parse_events({"events": [
            {"seq": 1, "type": "priority", "shard_id": "a", "priority": 9},
        ]})
        with self.assertRaises(SpecError) as ctx:
            validate_event_references(bad, spec)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.object_id, "event:1")


if __name__ == "__main__":
    unittest.main()
