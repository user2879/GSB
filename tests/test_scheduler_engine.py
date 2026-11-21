import json
import os
import tempfile
import unittest

from graphdiag.core.engine import Engine, result_to_dict
from graphdiag.core.errors import ReplayError, ScaleError
from graphdiag.core.fixedpoint import Quantizer
from graphdiag.core.loader import load_declaration
from graphdiag.core.models import content_digest
from graphdiag.core.scheduler import TraversalBudget, schedule_all

TINY = "data/tiny_graph.json"


class SchedulerTest(unittest.TestCase):
    def test_shared_budget_blocks_queued_queries(self):
        decl = load_declaration(TINY)
        qz = Quantizer(0)
        budget = TraversalBudget(3)
        report = schedule_all(
            decl.graph, decl.queries, qz, budget,
            max_workers=3, wait_timeout=0.1,
        )
        total_visited = sum(s.visited_nodes for s in report.ordered())
        exhausted = [s.query_id for s in report.ordered() if s.budget_exhausted]
        self.assertLessEqual(total_visited, 3)
        self.assertTrue(exhausted)
        # No query may bypass the shared budget.
        self.assertEqual(
            sum(s.budget_consumed for s in report.ordered()),
            budget.consumed(),
        )

    def test_schedule_records_monotonic_timestamps(self):
        decl = load_declaration(TINY)
        report = schedule_all(
            decl.graph, decl.queries, Quantizer(0),
            TraversalBudget(1000), max_workers=2,
        )
        for schedule in report.ordered():
            self.assertIsNotNone(schedule.start_monotonic_ms)
            self.assertIsNotNone(schedule.end_monotonic_ms)
            self.assertGreaterEqual(
                schedule.end_monotonic_ms, schedule.start_monotonic_ms
            )
            self.assertEqual(
                schedule.visited_nodes, schedule.result.budget_visited
            )


class EngineVersioningTest(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = Engine(self.db)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.engine.store.close()
        os.unlink(self.db)

    def test_staleness_and_whole_view_diff(self):
        self.engine.load(TINY)
        self.engine.run(budget_limit=1000)

        with open(TINY, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["edges"][0]["weight_micro"] = 3_500_000
        modified = tempfile.mktemp(suffix=".json")
        with open(modified, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            info = self.engine.load(modified)
            self.assertEqual(info["graph_version"], 2)
            self.assertIsNone(self.engine.current_view)  # stale, not displayed
            self.engine.run(budget_limit=1000)
            changes = self.engine.diff_views()
            self.assertTrue(changes)
            self.assertTrue(
                all(c.distance_delta_micro is not None for c in changes)
            )
        finally:
            os.unlink(modified)

    def test_query_only_change_bumps_query_version(self):
        self.engine.load(TINY)
        with open(TINY, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["queries"][0]["cap_micro"] = 8_000_000
        modified = tempfile.mktemp(suffix=".json")
        with open(modified, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            info = self.engine.load(modified)
            self.assertEqual(info["graph_version"], 1)
            self.assertEqual(info["query_version"], 2)
        finally:
            os.unlink(modified)


class SnapshotReplayTest(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = Engine(self.db)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.engine.store.close()
        os.unlink(self.db)

    def test_snapshot_restore_replay_ok(self):
        self.engine.load(TINY)
        self.engine.run(budget_limit=1000)
        self.engine.save_snapshot("snap")
        outcome = self.engine.restore_snapshot("snap")
        self.assertEqual(outcome["replay"], "ok")
        ledger = self.engine.store.read_ledger()
        self.assertTrue(ledger)

    def test_replay_divergence_raises_e_replay_with_location(self):
        self.engine.load(TINY)
        self.engine.run(budget_limit=1000)
        self.engine.save_snapshot("snap")
        row = self.engine.store.conn.execute(
            "SELECT payload_json FROM snapshots WHERE snapshot_id='snap'"
        ).fetchone()
        payload = json.loads(row[0])
        first_query = payload["conclusions"]["order"][0]
        target_key = next(
            iter(payload["conclusions"]["schedules"][first_query]["result"]["targets"])
        )
        payload["conclusions"]["schedules"][first_query]["result"]["targets"][
            target_key
        ]["dist_fp"] += 1
        body = {k: v for k, v in payload.items() if k != "integer_format_version"}
        self.engine.store.conn.execute(
            "UPDATE snapshots SET content_digest=?, payload_json=? WHERE snapshot_id='snap'",
            (content_digest(body), json.dumps(payload, sort_keys=True,
                                               separators=(",", ":"))),
        )
        self.engine.store.conn.commit()
        with self.assertRaises(ReplayError) as ctx:
            self.engine.restore_snapshot("snap")
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.detail["query_id"], first_query)
        self.assertEqual(ctx.exception.detail["target"], int(target_key))
        self.assertEqual(ctx.exception.detail["field"], "dist_fp")

    def test_scale_error_propagates(self):
        from graphdiag.core.fixedpoint import FP_MAX
        from graphdiag.core.models import Edge, Graph, GraphDeclaration, Query
        # Build an edge whose scale-6 quantization reaches the 64-bit ceiling.
        huge_micro = (FP_MAX // 10 ** 6 + 1) * 1_000_000
        graph = Graph(
            nodes=load_declaration(TINY).graph.nodes,
            edges=(Edge(0, 1, huge_micro),),
            adjacency={0: (Edge(0, 1, huge_micro),)},
            node_ids=frozenset({0, 1}),
        )
        declaration = GraphDeclaration(
            graph=graph,
            queries=(Query("q-huge", 0, (1,), huge_micro * 2),),
            scale_exponent=6,
        )
        self.engine.declaration = declaration
        self.engine.scale_exponent = 6
        with self.assertRaises(ScaleError) as ctx:
            self.engine.run(budget_limit=1000)
        self.assertEqual(ctx.exception.code, "E_SCALE")


if __name__ == "__main__":
    unittest.main()
