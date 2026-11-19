import unittest

from tactical_pathbench.workbench import build_workbench
from tactical_pathbench.errors import QueryUnreachableError, UnitInvalidError
from tactical_pathbench.astar import AStarSolver


class SolverTests(unittest.TestCase):
    def test_astar_and_flowfield_routes_match(self):
        wb = build_workbench(seed=20261119)
        for query in wb.sample.queries:
            dual = wb.solve_dual(query)
            self.assertEqual(dual["astar"].cost, dual["flowfield"].cost, query.query_id)
            self.assertEqual(tuple(dual["astar"].cells), tuple(dual["flowfield"].cells), query.query_id)
            self.assertGreater(dual["astar"].expanded, 0)

    def test_cache_hit_source_recorded(self):
        wb = build_workbench(seed=4)
        q = wb.sample.queries[0]
        wb.solve_query(q)
        wb.solve_query(q)
        self.assertEqual(wb.reuse_sources[q.key()], "path_cache:hit")

    def test_unreachable_and_invalid_unit(self):
        wb = build_workbench(seed=4)
        target = (0, 0)
        wb.grid.set_passable(target, False)
        with self.assertRaises(UnitInvalidError):
            AStarSolver(wb.grid).solve(target, (1, 1))
        source = wb.sample.queries[0].source
        self.assertNotEqual(source, target)
        with self.assertRaises(QueryUnreachableError):
            AStarSolver(wb.grid).solve(source, target)

    def test_unit_on_impassable_cell_rejected(self):
        wb = build_workbench(seed=4)
        unit = wb.sample.units[0]
        cell = unit.cell
        with self.assertRaises(UnitInvalidError):
            wb.move_unit(unit.unit_id, cell) if False else None
            wb.grid.set_passable(cell, False)
            from tactical_pathbench.generate import Unit as GenUnit
            bad = GenUnit(unit.unit_id, unit.faction, cell, unit.movement_budget)
            from tactical_pathbench.generate import _validate_units
            _validate_units(wb.grid, [bad])


if __name__ == "__main__":
    unittest.main()
