from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from planner_app.engine import PlanEngine
from planner_app.errors import AppError


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class EngineConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.engine = PlanEngine(
            WORKSPACE,
            ledger_path=base / "ledger.sqlite",
            snapshot_dir=base / "snapshots",
            log_path=base / "app.jsonl",
            budget_ms=1000,
        )

    def test_edit_is_incremental_but_byte_equal_to_full(self) -> None:
        before = self.engine.current_view("q_status")
        result = self.engine.submit_edits(
            [{"type": "prune_columns", "target": "project", "columns": ["min_amount", "max_amount"]}],
            query_name="q_status",
        )
        self.assertEqual(result["verification"]["status"], "MATCH")
        self.assertEqual(result["view"]["compile_status"], "READY")
        self.assertTrue(result["view"]["diff"]["summary_changed"])
        self.assertFalse(result["view"]["latency"].get("full_recompile_triggered", False))
        after = self.engine.current_view("q_status")
        self.assertEqual(after["plan_summary"], result["view"]["plan_summary"])
        self.assertNotEqual(before["plan_summary"], after["plan_summary"])

        unchanged = result["effects"][0]["unchanged_nodes"]
        self.assertTrue(unchanged)
        before_nodes = {node["node_id"]: node for node in before["nodes"]}
        after_nodes = {node["node_id"]: node for node in after["nodes"]}
        for entry in unchanged:
            node_id = entry["node_id"]
            self.assertTrue(entry["instance_preserved"])
            self.assertEqual(before_nodes[node_id]["summary"], after_nodes[node_id]["summary"])
            self.assertEqual(before_nodes[node_id]["version"], after_nodes[node_id]["version"])

    def test_background_full_verification_completes(self) -> None:
        result = self.engine.submit_edits(
            [{"type": "prune_columns", "target": "project", "columns": ["max_amount"]}],
            query_name="q_status",
            background_verify=True,
        )
        self.assertEqual(result["verification"]["status"], "PENDING")
        verification = self.engine.wait_for_verification(result["batch_id"], timeout=5)
        self.assertIsNotNone(verification)
        self.assertEqual(verification["status"], "MATCH")
        self.assertEqual(self.engine.current_view("q_status")["compile_status"], "READY")

    def test_parameter_change_is_monotonic_and_marks_old_view_stale(self) -> None:
        before = self.engine.current_view("q_status")
        self.engine.submit_edits(
            [{"type": "prune_columns", "target": "project", "columns": ["max_amount"]}],
            query_name="q_status",
        )
        edited_versions = dict(self.engine.versions)
        edited = self.engine.current_view("q_status")
        old_sql = self.engine.workspace.query_map()["q_status"].sql
        new_sql = old_sql.replace("amount >= 10", "amount >= 20")
        result = self.engine.update_query_sql("q_status", new_sql)
        self.assertTrue(result["replacement"])
        self.assertGreater(result["parameter_version"], before["parameter_version"])
        self.assertGreaterEqual(result["versions"]["edits"], edited_versions["edits"])
        self.assertGreater(result["versions"]["sql"], edited_versions["sql"])
        self.assertGreater(result["parameter_version"], edited["parameter_version"])
        after = self.engine.current_view("q_status")
        self.assertEqual(after["parameter_version"], result["parameter_version"])
        self.assertTrue(self.engine.stale_views)
        self.assertEqual(self.engine.stale_views[-1]["compile_status"], "STALE")
        self.assertTrue(self.engine.stale_views[-1]["stale"])
        self.assertGreaterEqual(self.engine.ledger.count_stale_batches(self.engine.workspace_id), 1)
        self.assertIn("q_status", result["diffs"])
        self.assertEqual(
            result["diffs"]["q_status"]["added"] + result["diffs"]["q_status"]["removed"]
            + result["diffs"]["q_status"]["summary_changed"],
            sorted(
                result["diffs"]["q_status"]["added"] + result["diffs"]["q_status"]["removed"]
                + result["diffs"]["q_status"]["summary_changed"],
                key=lambda value: value.encode("utf-8"),
            ),
        )

    def test_exported_workspace_restores_parameter_changes_with_same_ledger(self) -> None:
        old_sql = self.engine.workspace.query_map()["q_status"].sql
        new_sql = old_sql.replace("amount >= 10", "amount >= 20")
        result = self.engine.update_query_sql("q_status", new_sql)
        expected_summary = result["views"]["q_status"]["plan_summary"]
        exported = Path(self.temp.name) / "exported-workspace.json"
        self.engine.export_workspace(exported)
        restarted = PlanEngine(
            exported,
            ledger_path=Path(self.temp.name) / "ledger.sqlite",
            snapshot_dir=Path(self.temp.name) / "snapshots",
            log_path=Path(self.temp.name) / "app.jsonl",
            budget_ms=1000,
        )
        view = restarted.current_view("q_status")
        self.assertEqual(view["plan_summary"], expected_summary)
        self.assertIn("amount >= 20", view["sql"])

    def test_full_sample_edit_trace_matches(self) -> None:
        workspace = self.engine.workspace
        for batch in workspace.edit_batches:
            result = self.engine.submit_edits(list(batch.edits), query_name=batch.query)
            self.assertEqual(result["verification"]["status"], "MATCH")
        self.assertEqual(self.engine.replay_ledger()["status"], "MATCH")

    def test_quick_query_never_calls_full_compiler(self) -> None:
        with mock.patch.object(self.engine, "_compile_full", side_effect=AssertionError("full path used")):
            result = self.engine.quick_query("q_status")
        self.assertTrue(result["ok"])
        self.assertFalse(result["full_recompile_triggered"])
        self.assertEqual(result["fast_path"], "incremental_plan_cache_read")
        ledger_rows = self.engine.ledger.recent_latency(self.engine.workspace_id, 5)
        self.assertEqual(ledger_rows[0]["kind"], "interactive_query")
        self.assertEqual(ledger_rows[0]["latency_id"], result["latency_id"])
        self.assertEqual(ledger_rows[0]["truth_source"], "ledger:latency_records")

    def test_latency_over_budget_is_really_measured_and_recorded(self) -> None:
        base = Path(self.temp.name)
        engine = PlanEngine(
            WORKSPACE,
            ledger_path=base / "zero.sqlite",
            snapshot_dir=base / "zero-snapshots",
            log_path=base / "zero.jsonl",
            budget_ms=0,
        )
        result = engine.quick_query("q_status")
        self.assertEqual(result["warning_code"], "E_LATENCY")
        self.assertFalse(result["within_budget"])
        self.assertGreater(result["elapsed_ms"], 0)
        self.assertLess(result["mono_start_ns"], result["mono_end_ns"])
        row = engine.ledger.recent_latency(engine.workspace_id, 1)[0]
        self.assertEqual(row["latency_id"], result["latency_id"])
        self.assertEqual(row["code"], "E_LATENCY")
        self.assertEqual(row["budget_ms"], 0.0)

    def test_authoritative_recompute_matches_and_records(self) -> None:
        self.engine.submit_edits(
            [{"type": "push_filter", "target": "filter"}], query_name="q_status"
        )
        result = self.engine.authoritative_recompute("q_status")
        self.assertEqual(result["status"], "MATCH")
        self.assertIn("ledger:latency_records:", result["truth_source"])
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        verifications = self.engine.ledger.recent_verifications(self.engine.workspace_id, 5)
        self.assertTrue(any(item["mode"] == "AUTHORITATIVE" and item["status"] == "MATCH" for item in verifications))

    def test_authoritative_divergence_is_reported_and_located(self) -> None:
        plan = self.engine.states["q_status"].plan
        changed, _ = __import__("planner_app.plan", fromlist=["apply_edit"]).apply_edit(
            plan,
            {"type": "replace_scan_order", "target": "scan", "columns": ["order_day", "quantity", "status_code", "amount", "customer_id", "id"]},
            99,
        )
        with mock.patch.object(self.engine, "_compile_full", return_value=changed):
            with self.assertRaises(AppError) as raised:
                self.engine.authoritative_recompute("q_status")
        self.assertEqual(raised.exception.code, "E_DIVERGE")
        self.assertEqual(raised.exception.node_id, "scan")


if __name__ == "__main__":
    unittest.main()
