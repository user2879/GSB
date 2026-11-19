"""End-to-end solver, determinism and conflict/backtracking tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from level_solver.constraints import is_satisfied
from level_solver.errors import E_CONFLICT, SolverError
from level_solver.parser import parse_spec
from level_solver.solver import LevelSolver


def full_spec(**overrides):
    spec = {
        "width": 16,
        "height": 14,
        "seed": 20261119,
        "blocked": [[0, 0], [1, 0], [0, 1], [15, 13], [14, 13], [15, 12], [7, 6], [8, 6], [7, 7]],
        "concurrency": 4,
        "memory_budget": 8,
        "max_rollback": 12,
        "alternative_limit": 32,
        "constraints": [
            {"id": "rooms", "type": "min_rooms",
             "params": {"count": 5, "min_room_w": 2, "min_room_h": 2,
                        "max_room_w": 4, "max_room_h": 4}},
            {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
            {"id": "region_cap", "type": "max_region_area", "params": {"max_area": 120}},
            {"id": "all_connected", "type": "connectivity", "params": {}},
            {"id": "start_to_end", "type": "reachability",
             "params": {"start": [1, 1], "end": [14, 12]}},
        ],
    }
    spec.update(overrides)
    return spec


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lvl-solver-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dir(self, name):
        path = Path(self.tmp) / name
        return str(path)

    def test_all_constraints_satisfied(self):
        spec = parse_spec(full_spec())
        result = LevelSolver(spec, self._dir("a")).solve()
        for constraint in spec.constraints:
            ok, reason = is_satisfied(constraint, result.grid, result.rooms)
            self.assertTrue(ok, f"{constraint.cid}: {reason}")
        self.assertEqual(result.sequence, len(spec.constraints))
        self.assertEqual(sum(result.constraint_counts.values()), len(spec.constraints))

    def test_repeated_runs_cell_identical(self):
        spec_a = parse_spec(full_spec())
        result_a = LevelSolver(spec_a, self._dir("a")).solve()
        spec_b = parse_spec(full_spec())
        result_b = LevelSolver(spec_b, self._dir("b")).solve()
        self.assertEqual(result_a.grid.to_list(), result_b.grid.to_list())
        self.assertEqual(result_a.constraint_counts, result_b.constraint_counts)
        self.assertEqual([f.attempts for f in result_a.applied], [f.attempts for f in result_b.applied])

    def test_different_seed_can_change_layout(self):
        first = LevelSolver(parse_spec(full_spec()), self._dir("a")).solve()
        second = LevelSolver(parse_spec(full_spec(seed=20261120)), self._dir("b")).solve()
        self.assertNotEqual(first.grid.to_list(), second.grid.to_list())

    def test_conflict_raises_business_code_with_constraint_id(self):
        impossible = {
            "width": 10, "height": 10, "seed": 7, "blocked": [],
            "concurrency": 2, "memory_budget": 4, "max_rollback": 2,
            "alternative_limit": 16,
            "constraints": [
                {"id": "many_rooms", "type": "min_rooms",
                 "params": {"count": 4, "min_room_w": 3, "min_room_h": 3,
                            "max_room_w": 4, "max_room_h": 4}},
                {"id": "tiny_regions", "type": "max_region_area", "params": {"max_area": 8}},
                {"id": "joined", "type": "connectivity", "params": {}},
            ],
        }
        with self.assertRaises(SolverError) as ctx:
            LevelSolver(parse_spec(impossible), self._dir("c")).solve()
        self.assertEqual(ctx.exception.code, E_CONFLICT)
        self.assertIn("constraint", ctx.exception.details)
        self.assertIn("conflicts", ctx.exception.details)

    def test_no_partial_grid_output_on_conflict(self):
        impossible = {
            "width": 8, "height": 8, "seed": 1, "blocked": [],
            "max_rollback": 0, "alternative_limit": 8,
            "constraints": [
                {"id": "rooms", "type": "min_rooms",
                 "params": {"count": 3, "min_room_w": 3, "min_room_h": 3,
                            "max_room_w": 3, "max_room_h": 3}},
                {"id": "cap", "type": "max_region_area", "params": {"max_area": 4}},
                {"id": "joined", "type": "connectivity", "params": {}},
            ],
        }
        out_file = Path(self.tmp) / "result.json"
        with self.assertRaises(SolverError):
            solver = LevelSolver(parse_spec(impossible), self._dir("d"))
            solver.solve()
        self.assertFalse(out_file.exists())

    def test_rollback_recovers_from_spacing_conflict(self):
        spec_dict = {
            "width": 12, "height": 10, "seed": 11, "blocked": [],
            "concurrency": 2, "memory_budget": 4, "max_rollback": 16,
            "alternative_limit": 40,
            "constraints": [
                {"id": "rooms", "type": "min_rooms",
                 "params": {"count": 4, "min_room_w": 2, "min_room_h": 2,
                            "max_room_w": 3, "max_room_h": 3}},
                {"id": "gap", "type": "room_spacing", "params": {"min_distance": 2}},
                {"id": "joined", "type": "connectivity", "params": {}},
            ],
        }
        spec = parse_spec(spec_dict)
        result = LevelSolver(spec, self._dir("e")).solve()
        for constraint in spec.constraints:
            ok, reason = is_satisfied(constraint, result.grid, result.rooms)
            self.assertTrue(ok, f"{constraint.cid}: {reason}")

    def test_reachability_around_blocked_cluster(self):
        spec_dict = {
            "width": 12, "height": 10, "seed": 99,
            "blocked": [[5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6]],
            "concurrency": 3, "memory_budget": 6, "max_rollback": 8,
            "alternative_limit": 24,
            "constraints": [
                {"id": "rooms", "type": "min_rooms",
                 "params": {"count": 3, "min_room_w": 2, "min_room_h": 2,
                            "max_room_w": 3, "max_room_h": 3}},
                {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
                {"id": "path", "type": "reachability",
                 "params": {"start": [1, 1], "end": [10, 8]}},
            ],
        }
        spec = parse_spec(spec_dict)
        result = LevelSolver(spec, self._dir("f")).solve()
        ok, reason = is_satisfied(spec.constraints[-1], result.grid, result.rooms)
        self.assertTrue(ok, reason)

    def test_blocked_endpoint_precheck_conflict(self):
        spec_dict = {
            "width": 6, "height": 6, "seed": 1, "blocked": [[5, 5]],
            "max_rollback": 0, "alternative_limit": 8,
            "constraints": [
                {"id": "reach", "type": "reachability",
                 "params": {"start": [0, 0], "end": [5, 5]}},
            ],
        }
        with self.assertRaises(SolverError) as ctx:
            LevelSolver(parse_spec(spec_dict), self._dir("g")).solve()
        self.assertEqual(ctx.exception.code, E_CONFLICT)


if __name__ == "__main__":
    unittest.main()
