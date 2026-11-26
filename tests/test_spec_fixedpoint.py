from __future__ import annotations

import copy
import unittest

from gsb.errors import SpecError
from gsb.fixedpoint import (
    adjudicate_candidates,
    budget_boundary_view,
    global_water_error_bound,
)
from gsb.spec import parse_spec


def valid_spec() -> dict:
    return {
        "session_id": "unit",
        "scale_factor": 100,
        "budget": {"resource_units": 10},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 20101,
                "concurrency": 1,
                "lease_limit": 1,
                "seed": 7,
                "capacity_version": 1,
            }
        ],
        "shards": [
            {
                "shard_id": "s1",
                "dependencies": [],
                "rows": 8,
                "resource_units": 2,
                "priority": 1,
            }
        ],
    }


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec_has_stable_topo(self) -> None:
        spec = parse_spec(valid_spec())
        self.assertEqual(spec["topo_order"], ["s1"])
        self.assertEqual(spec["spec_digest"], spec["spec_digest"])

    def test_duplicate_node_id_reports_espec(self) -> None:
        raw = valid_spec()
        raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
        raw["nodes"][1]["port"] = 20102
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("nodes[1].node_id", ctx.exception.field or "")

    def test_duplicate_port_reports_location(self) -> None:
        raw = valid_spec()
        raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
        raw["nodes"][1]["node_id"] = "n2"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "nodes[1].port")

    def test_host_must_be_literal_loopback(self) -> None:
        raw = valid_spec()
        raw["nodes"][0]["host"] = "0.0.0.0"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "nodes[0].host")

    def test_missing_dependency(self) -> None:
        raw = valid_spec()
        raw["shards"][0]["dependencies"] = ["missing"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertIn("dependencies[0]", ctx.exception.field or "")

    def test_cycle_reports_espec(self) -> None:
        raw = valid_spec()
        raw["shards"].append(
            {
                "shard_id": "s2",
                "dependencies": ["s1"],
                "rows": 1,
                "resource_units": 1,
                "priority": 0,
            }
        )
        raw["shards"][0]["dependencies"] = ["s2"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertTrue(ctx.exception.details.get("cycle"))

    def test_positive_and_scale_rules(self) -> None:
        raw = valid_spec()
        raw["shards"][0]["resource_units"] = 0
        with self.assertRaises(SpecError):
            parse_spec(raw)

        raw = valid_spec()
        raw["scale_factor"] = 3
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "scale_factor")


class FixedPointTests(unittest.TestCase):
    def test_error_bound_segments_sum(self) -> None:
        bound = global_water_error_bound(6, 100)
        self.assertEqual(bound.truncation_q, 1)
        self.assertEqual(bound.accumulation_q, 5)
        self.assertEqual(bound.comparison_q, 1)
        self.assertEqual(bound.total_q, 7)

    def test_scale_changes_boundary_confidence(self) -> None:
        views = {
            scale: budget_boundary_view(
                used_units=29,
                demand_units=12,
                budget_units=40,
                scale=scale,
                active_term_count=2,
            )
            for scale in (10, 100, 1000)
        }
        self.assertTrue(views[10]["error_affected"])
        self.assertTrue(views[100]["error_affected"])
        self.assertFalse(views[1000]["error_affected"])
        self.assertEqual(views[10]["quantized_verdict"], "reject")
        self.assertTrue(views[10]["exact_feasible"] is False)

    def test_stable_topology_tie_is_marked_affected(self) -> None:
        result = adjudicate_candidates(
            [
                {"shard_id": "b", "topo_index": 1, "resource_units": 17},
                {"shard_id": "a", "topo_index": 0, "resource_units": 16},
            ],
            budget_units=100,
            scale=10,
        )
        self.assertEqual(result["winner"], "a")
        self.assertTrue(result["error_affected"])
        self.assertIn("b", result["ambiguous_with"])


if __name__ == "__main__":
    unittest.main()
