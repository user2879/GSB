from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_audit.diagnostics import analyze_state
from mesh_audit.mesh import MeshState
from mesh_audit.topology import build_topology
from tests.helpers import make_mesh, make_script, run_audit


class TopologyTests(unittest.TestCase):
    def test_defect_counts_and_rescan_consistency(self) -> None:
        script = make_script()
        state = MeshState.from_input(make_mesh(), script.fixed_scale)
        topology = build_topology(
            state.fixed, state.faces, state.scale, script.area_tolerance
        )
        summary = topology.summary()
        self.assertEqual(summary["vertex_count"], 10)
        self.assertEqual(summary["face_count"], 12)
        self.assertEqual(summary["edge_count"], 20)
        self.assertEqual(summary["nonmanifold_edge_count"], 1)
        self.assertEqual(summary["boundary_edge_count"], 6)
        self.assertEqual(summary["boundary_loop_count"], 1)
        self.assertEqual(summary["open_boundary_chain_count"], 1)
        self.assertEqual(summary["degenerate_face_count"], 1)
        self.assertEqual(summary["isolated_vertex_count"], 1)
        self.assertEqual(summary["orientation_conflict_edge_count"], 2)
        self.assertEqual(summary["euler_characteristic"], 2)
        self.assertTrue(summary["rescan_matches"])
        self.assertEqual(len(topology.vertex_faces), 10)
        self.assertEqual(len(topology.face_adjacency), 12)
        _, diagnostics = analyze_state(
            state, script.area_tolerance, script.concurrency, script.memory_budget_nodes
        )
        self.assertTrue(diagnostics["parallel_face_scan"]["mask_matches_indexed_pass"])

    def test_repair_changes_are_deterministic_and_degenerate_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = run_audit(Path(temporary), script=make_script())
            self.assertEqual(result.exit_code, 0)
            initial = result.report["summary"]["initial_diagnostics"]
            final = result.report["summary"]["final_diagnostics"]
            self.assertEqual(initial["degenerate_face_count"], 1)
            self.assertEqual(final["degenerate_face_count"], 0)
            self.assertEqual(final["orientation_conflict_edge_count"], 0)
            self.assertTrue(final["rescan_matches"])
            self.assertTrue(final["parallel_face_scan"]["mask_matches_indexed_pass"])
            self.assertEqual(result.report["steps"][1]["execution_count"], 1)
            self.assertEqual(result.report["steps"][2]["execution_count"], 1)
            self.assertEqual(result.report["steps"][3]["execution_count"], 1)
            self.assertEqual(result.report["steps"][4]["execution_count"], 16)
            hole_effect = result.report["steps"][1]["operator_effect"]
            smooth_effect = result.report["steps"][4]["operator_effect"]
            self.assertEqual(hole_effect["added_vertex_count"], 1)
            self.assertGreater(smooth_effect["changed_vertex_count"], 0)
            self.assertGreater(smooth_effect["max_abs_coordinate_delta"], 0.0)
            self.assertNotEqual(smooth_effect["curvature_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
