import unittest

from sharding_diag.errors import SpecError
from sharding_diag.model import parse_declaration, topological_order


def base_raw():
    return {
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 2,
             "lease_limit": 2, "seed": 1, "capacity_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 19002, "concurrency": 1,
             "lease_limit": 1, "seed": 2, "capacity_version": 1},
        ],
        "shards": [
            {"shard_id": "a", "rows": 10, "demand": 2, "priority": 1},
            {"shard_id": "b", "depends_on": ["a"], "rows": 10, "demand": 2, "priority": 1},
        ],
        "global": {"budget": 10, "freeze_threshold_ms": 100, "revoke_tolerance": 5},
    }


class SpecValidationTest(unittest.TestCase):
    def assert_reject(self, raw, object_id, field):
        with self.assertRaises(SpecError) as ctx:
            parse_declaration(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.object_id, object_id)
        self.assertEqual(ctx.exception.field, field)

    def test_valid_declaration_parses(self):
        decl = parse_declaration(base_raw())
        self.assertEqual(len(decl.nodes), 2)
        self.assertEqual(decl.global_spec.budget, 10)

    def test_duplicate_node_id(self):
        raw = base_raw()
        raw["nodes"][1]["node_id"] = "n1"
        raw["nodes"][1]["port"] = 19002
        self.assert_reject(raw, "n1", "node_id")

    def test_duplicate_shard_id(self):
        raw = base_raw()
        raw["shards"][1]["shard_id"] = "a"
        self.assert_reject(raw, "a", "shard_id")

    def test_non_loopback_host(self):
        raw = base_raw()
        raw["nodes"][0]["host"] = "10.0.0.1"
        self.assert_reject(raw, "n1", "host")

    def test_duplicate_port(self):
        raw = base_raw()
        raw["nodes"][1]["port"] = 19001
        self.assert_reject(raw, "n2", "port")

    def test_non_positive_concurrency(self):
        raw = base_raw()
        raw["nodes"][0]["concurrency"] = 0
        self.assert_reject(raw, "n1", "concurrency")

    def test_bool_rejected_as_integer(self):
        raw = base_raw()
        raw["nodes"][0]["lease_limit"] = True
        self.assert_reject(raw, "n1", "lease_limit")

    def test_unknown_dependency(self):
        raw = base_raw()
        raw["shards"][0]["depends_on"] = ["ghost"]
        self.assert_reject(raw, "a", "depends_on")

    def test_cycle_reports_first_node(self):
        raw = base_raw()
        raw["shards"][0]["depends_on"] = ["b"]
        with self.assertRaises(SpecError) as ctx:
            parse_declaration(raw)
        self.assertEqual(ctx.exception.field, "depends_on")
        self.assertIn("cycle", str(ctx.exception))

    def test_negative_demand_rejected(self):
        raw = base_raw()
        raw["shards"][0]["demand"] = -1
        self.assert_reject(raw, "a", "demand")

    def test_negative_tolerance_rejected(self):
        raw = base_raw()
        raw["global"]["revoke_tolerance"] = -1
        self.assert_reject(raw, "<global>", "revoke_tolerance")

    def test_topological_order_stable(self):
        order = topological_order(parse_declaration(base_raw()).shards)
        self.assertEqual(order, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
