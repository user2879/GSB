import unittest

from tactical_pathbench.workbench import build_workbench
from tactical_pathbench.errors import ClosureMismatchError
from tactical_pathbench.dependency import flow_id


class ClosureTests(unittest.TestCase):
    def _cells(self, wb):
        interior = boundary = None
        for cell in wb.grid._cell_region:
            if wb.grid.is_boundary_cell(cell):
                boundary = cell
            else:
                interior = cell
            if interior and boundary:
                break
        return interior, boundary

    def test_interior_change_invalidates_single_region_only(self):
        wb = build_workbench(seed=5)
        wb.solve_batch()
        interior, _ = self._cells(wb)
        record = wb.modify_terrain(interior, 6)
        closure = record.closure
        self.assertEqual(closure["invalidated_flow_regions"], [wb.grid.region_of_cell(interior)])
        self.assertEqual(closure["invalidated_jump_portals"], [])
        self.assertNotIn(
            flow_id(wb.grid.region_of_cell(interior)),
            [f"flow:R{r:03d}" for r in closure["reused_flows"]],
        )
        self.assertEqual(len(closure["reused_flows"]), len(wb.grid.region_ids()) - 1)

    def test_boundary_change_invalidates_jump_but_not_extra_flows(self):
        wb = build_workbench(seed=6)
        wb.solve_batch()
        _, boundary = self._cells(wb)
        record = wb.modify_terrain(boundary, 7)
        closure = record.closure
        self.assertEqual(closure["boundary_hits"], [[boundary[0], boundary[1]]])
        self.assertEqual(closure["interior_hits"], [])
        self.assertEqual(closure["invalidated_flow_regions"], [wb.grid.region_of_cell(boundary)])
        self.assertTrue(closure["invalidated_jump_portals"])
        # 跳点失效不反向污染其它区域流场
        self.assertEqual(len(closure["reused_flows"]), len(wb.grid.region_ids()) - 1)

    def test_incremental_matches_full_rebuild(self):
        wb = build_workbench(seed=8)
        wb.solve_batch()
        _, boundary = self._cells(wb)
        wb.modify_terrain(boundary, 4)
        report = wb.consistency_check()
        self.assertTrue(report.consistent, report.to_payload())

    def test_closure_mismatch_detection(self):
        wb = build_workbench(seed=8)
        wb.solve_batch()
        interior, _ = self._cells(wb)
        closure = wb.graph.tile_closure([interior])
        with self.assertRaises(ClosureMismatchError):
            wb.graph.assert_closure_precision(closure, {999})

    def test_jump_and_flow_direction_not_reversed(self):
        wb = build_workbench(seed=8)
        wb.solve_batch()
        _, boundary = self._cells(wb)
        before_flow = wb.flows[wb.grid.region_of_cell(boundary)].content_hash()
        closure = wb.graph.tile_closure([boundary])
        # 仅标记 jump 失效，不应用 flow 失效，flow 内容哈希应保持
        for portal in closure.invalidated_jump_portals:
            self.assertTrue(portal)
        self.assertEqual(wb.flows[wb.grid.region_of_cell(boundary)].content_hash(), before_flow)


if __name__ == "__main__":
    unittest.main()
