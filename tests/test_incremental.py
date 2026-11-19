from __future__ import annotations

import copy
import unittest

from forensic_workbench.config import AnalysisWindow, ScopedRect
from forensic_workbench.errors import ForensicError
from forensic_workbench.graph import compare_graphs
from forensic_workbench.pipeline import execute
from forensic_workbench.state import SnapshotStore
from tests.support import TemporaryWorkspace, make_workspace_source


class IncrementalTests(unittest.TestCase):
    def test_mask_change_closure_and_cross_gop_item_scope(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            execute(workspace, config, verify_full=True)
            base = SnapshotStore(workspace).load()
            self.assertIsNotNone(base)
            variant = config.with_regions(
                mask=list(config.watermark_mask.regions)
                + [ScopedRect(0, 0, 8, 8, frame_start=20, frame_end=30)]
            )
            result = execute(workspace, variant, verify_full=True)
            expected = {
                "conclusion",
                "gop:002",
                "xgop:001-002",
                "xgop:002-003",
                *{f"frame:{index:06d}" for index in range(20, 30)},
            }
            self.assertEqual(set(result.audit["recomputed_nodes"]), expected)
            self.assertTrue(result.full_verified)
            persisted = SnapshotStore(workspace).load()
            self.assertIsNotNone(persisted)
            self.assertEqual(
                len(persisted.graph.audit_history[-1]["reused_nodes"]),
                len(result.audit["reused_nodes"]),
            )
            self.assertEqual(
                result.graph.nodes["xgop:000-001"].content_hash,
                base.graph.nodes["xgop:000-001"].content_hash,
            )
            self.assertEqual(
                result.graph.nodes["xgop:003-004"].content_hash,
                base.graph.nodes["xgop:003-004"].content_hash,
            )
            self.assertNotEqual(
                result.graph.nodes["xgop:001-002"].content_hash,
                base.graph.nodes["xgop:001-002"].content_hash,
            )
            for index in range(20, 30):
                node_id = f"frame:{index:06d}"
                old_excluded = base.graph.nodes[node_id].payload["excluded_pixels"]
                new_excluded = result.graph.nodes[node_id].payload["excluded_pixels"]
                self.assertGreater(new_excluded, old_excluded)
            for index in range(0, 20):
                node_id = f"frame:{index:06d}"
                self.assertEqual(
                    result.graph.nodes[node_id].content_hash,
                    base.graph.nodes[node_id].content_hash,
                )

    def test_analysis_window_recomputes_only_changed_boundary_gops(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            execute(workspace, config, verify_full=True)
            variant = config.with_regions(window=AnalysisWindow(23, 47))
            result = execute(workspace, variant, verify_full=True)
            self.assertEqual(result.audit["active_gops"], [2, 3, 4])
            self.assertEqual(
                set(result.audit["recomputed_nodes"]),
                {
                    "conclusion",
                    "gop:002",
                    "gop:004",
                    "xgop:002-003",
                    "xgop:003-004",
                },
            )
            self.assertFalse(
                any(node.startswith("frame:") for node in result.audit["recomputed_nodes"])
            )
            self.assertTrue(result.full_verified)

    def test_sampling_scope_and_no_change_reuse(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            first = execute(workspace, config, verify_full=True)
            second = execute(workspace, config, verify_full=True)
            self.assertEqual(second.audit["recomputed_nodes"], [])
            self.assertEqual(first.conclusion_hash, second.conclusion_hash)
            self.assertTrue(second.full_verified)
            regio0 = ScopedRect(0, 0, source.spec.width, source.spec.height, 0, 40)
            regio1 = ScopedRect(2, 2, source.spec.width - 4, source.spec.height - 4, 40, 50)
            regio2 = ScopedRect(
                0, 0, source.spec.width, source.spec.height, 50, source.frame_count
            )
            variant = config.with_regions(sampling=[regio0, regio1, regio2])
            result = execute(workspace, variant, verify_full=True)
            expected = {
                "conclusion",
                "gop:004",
                "xgop:003-004",
                "xgop:004-005",
                *{f"frame:{index:06d}" for index in range(40, 50)},
            }
            self.assertEqual(set(result.audit["recomputed_nodes"]), expected)
            self.assertTrue(result.full_verified)

    def test_closure_mismatch_names_item_and_edges(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            execute(workspace, config, verify_full=True)
            graph = SnapshotStore(workspace).load().graph  # type: ignore[union-attr]
            tampered = copy.deepcopy(graph)
            cross_id = sorted(
                node_id
                for node_id, node in tampered.nodes.items()
                if node.node_type == "cross_gop"
            )[0]
            tampered.nodes[cross_id].payload["score"] = 0.987654
            tampered.nodes[cross_id].seal()
            with self.assertRaises(ForensicError) as raised:
                compare_graphs(tampered, graph, context={"test": "cross-item"})
            error = raised.exception
            self.assertEqual(error.code, "E_CLOSURE_MISMATCH")
            self.assertEqual(error.context["node_id"], cross_id)
            self.assertTrue(error.context["comparison_item"])
            self.assertTrue(error.context["invalidated_edges"])
            self.assertTrue(error.context["upstream_edges"])


if __name__ == "__main__":
    unittest.main()
