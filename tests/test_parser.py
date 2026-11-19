"""Validation and business-code tests for specification parsing."""

from __future__ import annotations

import unittest

from level_solver.errors import E_DOMAIN, E_SCHEMA, SolverError
from level_solver.parser import parse_spec


def base_spec(**overrides):
    spec = {
        "width": 8,
        "height": 8,
        "seed": 1,
        "blocked": [[0, 0]],
        "constraints": [
            {"id": "r", "type": "min_rooms", "params": {"count": 2,
             "min_room_w": 2, "min_room_h": 2, "max_room_w": 3, "max_room_h": 3}}
        ],
    }
    spec.update(overrides)
    return spec


class ParserTests(unittest.TestCase):
    def test_valid_spec_normalises(self):
        spec = parse_spec(base_spec())
        self.assertEqual(spec.width * spec.height - 1, spec.available_cells)
        self.assertEqual(spec.constraints[0].cid, "r")

    def test_non_positive_dimensions_rejected(self):
        with self.assertRaises(SolverError) as ctx:
            parse_spec(base_spec(width=0))
        self.assertEqual(ctx.exception.code, E_DOMAIN)
        with self.assertRaises(SolverError) as ctx:
            parse_spec(base_spec(height=-1))
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_blocked_cell_outside_grid(self):
        with self.assertRaises(SolverError) as ctx:
            parse_spec(base_spec(blocked=[[8, 0]]))
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_unknown_constraint_type(self):
        bad = base_spec(constraints=[{"id": "x", "type": "nope", "params": {}}])
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, "E_CONSTRAINT")

    def test_negative_spacing_rejected(self):
        bad = base_spec(
            constraints=[{"id": "s", "type": "room_spacing", "params": {"min_distance": -1}}]
        )
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_min_rooms_exceeds_available_cells(self):
        bad = base_spec(
            width=3,
            height=3,
            blocked=[],
            constraints=[
                {"id": "r", "type": "min_rooms",
                 "params": {"count": 10, "min_room_w": 1, "min_room_h": 1,
                            "max_room_w": 1, "max_room_h": 1}}
            ],
        )
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_missing_seed_rejected(self):
        spec = base_spec()
        del spec["seed"]
        with self.assertRaises(SolverError) as ctx:
            parse_spec(spec)
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_non_finite_and_wrong_type_rejected(self):
        bad = base_spec(seed=float("nan"))
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, E_DOMAIN)
        bad = base_spec(width="wide")
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_duplicate_constraint_id_rejected(self):
        bad = base_spec(
            constraints=[
                {"id": "r", "type": "room_spacing", "params": {"min_distance": 0}},
                {"id": "r", "type": "connectivity", "params": {}},
            ]
        )
        with self.assertRaises(SolverError) as ctx:
            parse_spec(bad)
        self.assertEqual(ctx.exception.code, E_DOMAIN)

    def test_schema_must_be_object(self):
        with self.assertRaises(SolverError) as ctx:
            parse_spec([])
        self.assertEqual(ctx.exception.code, E_SCHEMA)

    def test_memory_budget_below_concurrency_rejected(self):
        with self.assertRaises(SolverError) as ctx:
            parse_spec(base_spec(concurrency=8, memory_budget=2))
        self.assertEqual(ctx.exception.code, E_DOMAIN)


if __name__ == "__main__":
    unittest.main()
