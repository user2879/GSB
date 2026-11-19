from __future__ import annotations

import unittest
from dataclasses import replace

from tactical_path.consistency import compare_incremental_with_full
from tactical_path.demo import _choose_edit_cell
from tactical_path.engine import FlowEntry, MapEdit, TierEngine, path_node_id
from tactical_path.errors import ErrorCode, TacticalError
from tactical_path.strategy import StrategyPlanner

from .helpers import bundle, config


class IncrementalTests(unittest.TestCase):
    def _solved_engine(self):
        cfg = config()
        loaded = bundle("small")
        planner = StrategyPlanner(cfg)
        plan = planner.plan(
            loaded.map_data, loaded.queries, cfg.budgets["small"]
        )
        engine = TierEngine(loaded, cfg)
        result = engine.solve_batch(
            loaded.queries, plan.decisions, cfg.budgets["small"]
        )
        return cfg, loaded, planner, plan, engine, result

    def test_closure_equals_recompute_and_outside_nodes_are_reused(self) -> None:
        cfg, loaded, _planner, plan, engine, result = self._solved_engine()
        cell, new_terrain = _choose_edit_cell(loaded, result["routes"])
        before_versions = {
            node_id: node.version
            for node_id, node in engine.graph.nodes.items()
            if node.kind == "path"
        }
        incremental = engine.apply_edits(
            [MapEdit(cell=cell, terrain=new_terrain)], plan.decisions
        )
        self.assertEqual(
            set(incremental["closure"]), set(incremental["recomputed"])
        )
        self.assertGreater(len(incremental["reused_nodes"]), 0)
        closure_paths = {
            node_id
            for node_id in incremental["closure"]
            if engine.graph.nodes[node_id].kind == "path"
        }
        reused_paths = {
            node_id
            for node_id in before_versions
            if node_id not in closure_paths
        }
        self.assertTrue(reused_paths)
        for node_id in reused_paths:
            self.assertEqual(before_versions[node_id], engine.graph.nodes[node_id].version)

    def test_incremental_and_full_rebuild_match_on_cost_and_sequence(self) -> None:
        cfg, loaded, _planner, plan, engine, result = self._solved_engine()
        cell, new_terrain = _choose_edit_cell(loaded, result["routes"])
        incremental = engine.apply_edits(
            [MapEdit(cell=cell, terrain=new_terrain)], plan.decisions
        )
        report = compare_incremental_with_full(
            engine,
            incremental["routes"],
            plan.decisions,
            cfg,
            affected_regions=incremental["affected_regions"],
        )
        self.assertTrue(report["all_consistent"])
        self.assertEqual(report["max_cost_deviation"], 0)
        self.assertIsNone(report["first_divergence"])
        self.assertEqual(
            report["incremental_route_hashes"],
            report["full_rebuild_route_hashes"],
        )

    def test_flow_invalidation_cascades_to_path_cache(self) -> None:
        cfg, loaded, _planner, plan, engine, result = self._solved_engine()
        cell, new_terrain = _choose_edit_cell(loaded, result["routes"])
        incremental = engine.apply_edits(
            [MapEdit(cell=cell, terrain=new_terrain)], plan.decisions
        )
        closure_kinds = {
            engine.graph.nodes[node_id].kind
            for node_id in incremental["closure"]
        }
        self.assertIn("flow", closure_kinds)
        self.assertIn("path", closure_kinds)

    def test_closure_mismatch_raises_business_error(self) -> None:
        cfg, loaded, _planner, plan, engine, result = self._solved_engine()
        cell, new_terrain = _choose_edit_cell(loaded, result["routes"])
        original = engine._ensure_flow

        def broken_flow(region_id, target, mobility, *, force=False):
            entry = original(region_id, target, mobility, force=force)
            return FlowEntry(
                field=entry.field,
                node_id="flow:forced-mismatch",
                deps_versions=entry.deps_versions,
            )

        engine._ensure_flow = broken_flow  # type: ignore[method-assign]
        with self.assertRaises(TacticalError) as raised:
            engine.apply_edits(
                [MapEdit(cell=cell, terrain=new_terrain)], plan.decisions
            )
        self.assertEqual(raised.exception.code, ErrorCode.CLOSURE_MISMATCH)

    def test_terrain_range_error(self) -> None:
        _cfg, _loaded, _planner, plan, engine, _result = self._solved_engine()
        with self.assertRaises(TacticalError) as raised:
            engine.apply_edits([MapEdit(cell=0, terrain=0)], plan.decisions)
        self.assertEqual(raised.exception.code, ErrorCode.TERRAIN_RANGE)

