import unittest

from shardcast.errors import DivergeError
from shardcast.propagation import (
    assert_rebuild_parity,
    compare_field_by_field,
    plan_invalidation,
)
from shardcast.spec import parse_spec


def base_raw() -> dict:
    return {
        "format_version": 1,
        "lease_ttl_ms": 1000,
        "global_budget": {1: 10},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 19601,
                "seed": 1,
                "initial_capacity_version": 1,
                "capacities": [{"version": 1, "concurrency": 2, "lease_limit": 2}],
            }
        ],
        "shards": [
            {"shard_id": "a", "depends_on": [], "sample_rows": 10,
             "resource_units": 1, "priority": 0},
            {"shard_id": "b", "depends_on": ["a"], "sample_rows": 10,
             "resource_units": 1, "priority": 0},
            {"shard_id": "c", "depends_on": ["a"], "sample_rows": 10,
             "resource_units": 1, "priority": 0},
            {"shard_id": "d", "depends_on": ["b", "c"], "sample_rows": 10,
             "resource_units": 1, "priority": 0},
        ],
        "events": [{"kind": "arrive", "shard_id": "a"}],
    }


class PropagationTests(unittest.TestCase):
    def test_resource_change_invalidates_dependents_in_topo_order(self) -> None:
        old = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["shards"][0]["resource_units"] = 5
        new = parse_spec(new_raw)
        steps = plan_invalidation(old, new)
        ids = [step["shard_id"] for step in steps]
        self.assertEqual(ids, ["a", "b", "c", "d"])
        self.assertEqual(steps[0]["reason"], "resource_units_changed")
        self.assertEqual(steps[1]["reason"], "upstream_invalid")
        self.assertEqual(steps[1]["caused_by"], "a")

    def test_sample_row_change_only_invalidates_descendants(self) -> None:
        old = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["shards"][2]["sample_rows"] = 22
        new = parse_spec(new_raw)
        steps = plan_invalidation(old, new)
        ids = {step["shard_id"] for step in steps}
        self.assertEqual(ids, {"c", "d"})

    def test_capacity_version_change_invalidates_all_shards(self) -> None:
        old = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["global_budget"] = {1: 10, 2: 8}
        new_raw["nodes"][0]["capacities"].append(
            {"version": 2, "concurrency": 1, "lease_limit": 1}
        )
        new = parse_spec(new_raw)
        steps = plan_invalidation(old, new)
        self.assertEqual({step["shard_id"] for step in steps}, {"a", "b", "c", "d"})

    def test_unaffected_shards_keep_byte_identical_state(self) -> None:
        old = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["shards"][2]["sample_rows"] = 33
        new = parse_spec(new_raw)
        steps = plan_invalidation(old, new)
        invalidated = [step["shard_id"] for step in steps]
        unaffected = sorted(set(old.shard_ids) - set(invalidated))
        self.assertEqual(unaffected, ["a", "b"])

    def test_field_by_field_parity_checker_passes_and_locates_diff(self) -> None:
        left = {"summaries": {"a": {"x": 1}}, "ownership": {"a": "n1"},
                "water_marks": {"n1": {"shard_count": 1}},
                "propagation_basis": [{"shard_id": "a"}], "invalidated": ["a"]}
        right = dict(left)
        assert_rebuild_parity(
            left,
            {
                "summaries": {"a": {"x": 1}},
                "ownership": {"a": "n1"},
                "water_marks": {"n1": {"shard_count": 1}},
                "propagation_basis": [{"shard_id": "a"}],
                "invalidated": ["a"],
            },
            invalidated=["a"],
        )
        divergent = dict(left)
        divergent["summaries"] = {"a": {"x": 2}}
        with self.assertRaises(DivergeError) as caught:
            compare_field_by_field(left, divergent, context="rebuild")
        self.assertEqual(caught.exception.code, "E_DIVERGE")
        self.assertIn("summaries.a.x", caught.exception.path)


if __name__ == "__main__":
    unittest.main()
