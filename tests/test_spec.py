import json
import unittest
from pathlib import Path

from shardcast.errors import SpecError
from shardcast.spec import load_spec, parse_spec, topological_order


def valid_raw() -> dict:
    return {
        "format_version": 1,
        "lease_ttl_ms": 250,
        "global_budget": {1: 10, 2: 8},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 19301,
                "seed": 7,
                "initial_capacity_version": 1,
                "capacities": [
                    {"version": 1, "concurrency": 2, "lease_limit": 3},
                    {"version": 2, "concurrency": 1, "lease_limit": 2},
                ],
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": 19302,
                "seed": 9,
                "initial_capacity_version": 1,
                "capacities": [
                    {"version": 1, "concurrency": 1, "lease_limit": 1},
                    {"version": 2, "concurrency": 2, "lease_limit": 2},
                ],
            },
        ],
        "shards": [
            {"shard_id": "a", "depends_on": [], "sample_rows": 10,
             "resource_units": 1, "priority": 0},
            {"shard_id": "b", "depends_on": ["a"], "sample_rows": 12,
             "resource_units": 2, "priority": 1},
            {"shard_id": "c", "depends_on": ["a"], "sample_rows": 8,
             "resource_units": 1, "priority": 2},
        ],
        "events": [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "capacity", "capacity_version": 2},
        ],
    }


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec_parses(self) -> None:
        spec = parse_spec(valid_raw())
        self.assertEqual(spec.node_ids, ("n1", "n2"))
        self.assertEqual(spec.global_budget, {1: 10, 2: 8})
        self.assertEqual(topological_order(spec), ("a", "b", "c"))

    def test_duplicate_node_id_reports_path(self) -> None:
        raw = valid_raw()
        raw["nodes"][1]["node_id"] = "n1"
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("nodes[1].node_id", caught.exception.path)

    def test_non_loopback_host_rejected_before_anything_runs(self) -> None:
        raw = valid_raw()
        raw["nodes"][0]["host"] = "10.0.0.1"
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "nodes[0].host")

    def test_duplicate_port_rejected(self) -> None:
        raw = valid_raw()
        raw["nodes"][1]["port"] = 19301
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "nodes[1].port")

    def test_port_range_rejected(self) -> None:
        raw = valid_raw()
        raw["nodes"][0]["port"] = 70000
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "nodes[0].port")

    def test_negative_quota_rejected(self) -> None:
        raw = valid_raw()
        raw["nodes"][0]["capacities"][0]["concurrency"] = -1
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "nodes[0].capacities[0].concurrency")

    def test_zero_sample_rows_rejected(self) -> None:
        raw = valid_raw()
        raw["shards"][0]["sample_rows"] = 0
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "shards[0].sample_rows")

    def test_unknown_dependency_rejected(self) -> None:
        raw = valid_raw()
        raw["shards"][0]["depends_on"] = ["missing"]
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "shards[0].depends_on")

    def test_dependency_cycle_rejected(self) -> None:
        raw = valid_raw()
        raw["shards"][0]["depends_on"] = ["b"]
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("cycle", caught.exception.message)

    def test_event_unknown_shard_rejected(self) -> None:
        raw = valid_raw()
        raw["events"][0]["shard_id"] = "nope"
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.path, "events[0].shard_id")

    def test_load_from_json_file(self) -> None:
        tmp = Path(__file__).with_name("tmp-spec.json")
        tmp.write_text(json.dumps(valid_raw()), encoding="utf-8")
        try:
            spec = load_spec(tmp)
            self.assertEqual(len(spec.shards), 3)
        finally:
            tmp.unlink()

    def test_content_digest_stable(self) -> None:
        first = parse_spec(valid_raw()).content_digest()
        second = parse_spec(valid_raw()).content_digest()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
