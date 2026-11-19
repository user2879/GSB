from __future__ import annotations

import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

from cdn_edge.errors import PipelineError
from cdn_edge.fixed import FixedGrid, parse_grid_power, stable_descending_rank
from cdn_edge.spec import load_spec, parse_spec

from tests.helpers import sample_raw


class FixedPointTests(unittest.TestCase):
    def test_floor_quantization_and_bounds(self) -> None:
        grid = FixedGrid(3)
        self.assertEqual(grid.q_ratio(2, 3), 666)
        self.assertEqual(grid.q_average_ms(2500, 2), 1250)
        self.assertEqual(grid.single_value_error_bound_units, 1)
        self.assertEqual(grid.ratio_comparison_error_bound_units, 2)
        self.assertTrue(grid.compare_with_bound(500, 498)["within_error_bound"])
        self.assertFalse(grid.compare_with_bound(500, 497)["within_error_bound"])
        self.assertEqual(grid.truncation, "floor_toward_negative_infinity")

    def test_stable_rank_is_input_order_independent(self) -> None:
        grid = FixedGrid(3)
        left = stable_descending_rank([("b", 500), ("a", 500), ("c", 100)], grid)
        right = stable_descending_rank([("c", 100), ("a", 500), ("b", 500)], grid)
        self.assertEqual(left, right)
        self.assertEqual([item["name"] for item in left[:2]], ["a", "b"])
        self.assertTrue(left[0]["error_bound_dominated"])


class SpecValidationTests(unittest.TestCase):
    def test_packaged_sample_matches_repository_sample(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            (root / "samples" / "sample_spec.json").read_bytes(),
            (root / "cdn_edge" / "data" / "sample_spec.json").read_bytes(),
        )

    def test_sample_spec_is_valid(self) -> None:
        spec = load_spec(Path(__file__).resolve().parent.parent / "samples" / "sample_spec.json")
        self.assertEqual(spec.grid.power, 3)
        self.assertEqual(len(spec.nodes), 3)
        self.assertEqual(len(spec.requests), 18)
        self.assertIn("/all.js", spec.content)

    def test_cycle_reports_e_cycle_with_path_and_field(self) -> None:
        raw = sample_raw()
        for item in raw["content"]:
            if item["path"] == "/catalog/base.json":
                item["derivation"] = {"inputs": [{"path": "/all.js", "coefficient": 1}]}
        with self.assertRaises(PipelineError) as raised:
            parse_spec(raw)
        error = raised.exception
        self.assertEqual(error.code, "E_CYCLE")
        self.assertIn("path", error.to_dict())
        self.assertEqual(error.field, "inputs")
        self.assertTrue(error.details["spec_rejected"])
        self.assertEqual(error.details["cycle"][0], error.details["cycle"][-1])

    def test_non_positive_ttl_is_e_spec(self) -> None:
        raw = sample_raw()
        raw["content"][0]["ttl_seconds"] = 0
        with self.assertRaises(PipelineError) as raised:
            parse_spec(raw)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.path, "content[0]")
        self.assertEqual(raised.exception.field, "ttl_seconds")

    def test_duplicate_cache_key_field_is_e_spec(self) -> None:
        raw = sample_raw()
        raw["nodes"][0]["cache_key_fields"] = ["path", "path"]
        with self.assertRaises(PipelineError) as raised:
            parse_spec(raw)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.path, "nodes[0].cache_key_fields[1]")
        self.assertEqual(raised.exception.field, "$")

    def test_unknown_dependency_is_e_spec(self) -> None:
        raw = sample_raw()
        raw["content"][0]["derivation"] = {"inputs": [{"path": "/missing", "coefficient": 2}]}
        with self.assertRaises(PipelineError) as raised:
            parse_spec(raw)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.field, "path")
        self.assertEqual(raised.exception.details["referenced_path"], "/missing")


    def test_source_update_cycle_is_e_cycle(self) -> None:
        raw = sample_raw()
        raw["source_updates"] = [
            {
                "at_ms": 10,
                "path": "/catalog/base.json",
                "derivation": {"inputs": [{"path": "/all.js", "coefficient": 1}]},
            }
        ]
        with self.assertRaises(PipelineError) as raised:
            parse_spec(raw)
        self.assertEqual(raised.exception.code, "E_CYCLE")
        self.assertIn("source_updates[0]", raised.exception.path)

    def test_grid_is_e_grid(self) -> None:
        with self.assertRaises(PipelineError) as raised:
            parse_grid_power("1.5")
        self.assertEqual(raised.exception.code, "E_GRID")
        self.assertEqual(raised.exception.field, "power")
        with self.assertRaises(PipelineError) as raised:
            FixedGrid(-1)
        self.assertEqual(raised.exception.code, "E_GRID")


if __name__ == "__main__":
    unittest.main()
