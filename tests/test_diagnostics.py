from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slowquery_workbench import Workbench
from slowquery_workbench.diagnostics import Candidate

from tests.helpers import SAMPLE_QUERY, new_sample_workbench


class DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_four_candidate_classes_replay_and_choose_fastest(self) -> None:
        workbench = new_sample_workbench(self.root)
        result = workbench.diagnose(
            SAMPLE_QUERY,
            timeout_ms=500,
            budget_ms=5000,
            threshold_ms=0,
            threshold_rows=0,
        )
        self.assertTrue(result["slow"])
        kinds = {item["kind"] for item in result["candidates"]}
        self.assertEqual(
            kinds,
            {"predicate_merge", "index_hint", "projection_convergence", "join_order"},
        )
        valid = [item for item in result["candidates"] if item["status"] == "valid"]
        self.assertEqual(len(valid), 4)
        for candidate in valid:
            self.assertEqual(candidate["result_rows"], result["original"]["result_rows"])
            self.assertGreater(candidate["elapsed_ms"], 0)
            self.assertGreater(candidate["index_version"], 0)
            self.assertGreater(candidate["stats_version"], 0)
            self.assertTrue(candidate["rationale"])
        fastest = min(valid, key=lambda item: (item["elapsed_ms"], item["candidate_id"]))
        self.assertEqual(result["conclusion"]["optimal_candidate_id"], fastest["candidate_id"])
        self.assertEqual(len(workbench.conclusions()), 1)
        workbench.close()

    def test_inequivalent_candidate_is_invalidated_and_not_selected(self) -> None:
        workbench = new_sample_workbench(self.root)
        original_sql = (
            "SELECT id, amount FROM orders WHERE amount >= 100 AND amount >= 200"
        )
        bad_query_sql = "SELECT id, amount FROM orders WHERE amount < 0"
        from slowquery_workbench.query import parse_select

        parsed = parse_select(original_sql)
        bad = Candidate(
            candidate_id="c_bad_equiv_test",
            query_id=parsed.query_id,
            kind="projection_convergence",
            sql=bad_query_sql,
            rationale="test-only deliberately inequivalent candidate",
            referenced_indexes=(),
            index_version=workbench.versions()["index_version"],
            stats_version=workbench.versions()["stats_version"],
        )
        with mock.patch("slowquery_workbench.diagnostics.generate_candidates", return_value=[bad]):
            result = workbench.diagnose(
                original_sql,
                timeout_ms=500,
                budget_ms=3000,
                threshold_ms=0,
                threshold_rows=0,
            )
        self.assertEqual(result["candidates"][0]["status"], "invalid_equiv")
        self.assertEqual(result["candidates"][0]["error_code"], "E_EQUIV")
        self.assertIn("E_EQUIV", workbench.logger.path.read_text(encoding="utf-8"))
        self.assertIsNone(result["conclusion"]["optimal_candidate_id"])
        self.assertIsNone(workbench.current_best(result["query_id"])["optimal_candidate_id"])
        workbench.close()

    def test_index_change_marks_only_affected_conclusions_stale(self) -> None:
        workbench = new_sample_workbench(self.root)
        workbench.create_table(
            "CREATE TABLE audit (id INTEGER PRIMARY KEY, action TEXT NOT NULL, code INTEGER NOT NULL)"
        )
        workbench.append_rows(
            "audit",
            ["id", "action", "code"],
            [(number, "login" if number % 2 else "logout", number) for number in range(1, 201)],
        )
        audit_sql = "SELECT id, action FROM audit WHERE action = 'login'"
        audit_result = workbench.diagnose(audit_sql, timeout_ms=500, budget_ms=3000, threshold_ms=0, threshold_rows=0)
        order_result = workbench.diagnose(SAMPLE_QUERY, timeout_ms=500, budget_ms=5000, threshold_ms=0, threshold_rows=0)
        audit_id = audit_result["query_id"]
        order_id = order_result["query_id"]
        audit_before = workbench.session.conclusion(audit_id)
        order_before = workbench.session.conclusion(order_id)
        self.assertFalse(audit_before["is_stale"])
        self.assertFalse(order_before["is_stale"])

        workbench.add_index("orders", ["amount"], name="idx_orders_amount_invalidation")
        event = workbench.change_events()[-1]
        self.assertEqual(event["source_version_field"], "index_version")
        self.assertEqual(event["affected_query_ids"], [order_id])
        self.assertEqual(event["affected_tables"], ["orders"])
        audit_after = workbench.session.conclusion(audit_id)
        order_after = workbench.session.conclusion(order_id)
        self.assertFalse(audit_after["is_stale"])
        self.assertEqual(audit_after["updated_at"], audit_before["updated_at"])
        self.assertTrue(order_after["is_stale"])
        self.assertIsNone(workbench.current_best(order_id))
        self.assertIsNotNone(workbench.current_best(audit_id))
        workbench.close()

    def test_data_and_statistics_changes_propagate(self) -> None:
        workbench = new_sample_workbench(self.root)
        result = workbench.diagnose(SAMPLE_QUERY, timeout_ms=500, budget_ms=5000, threshold_ms=0, threshold_rows=0)
        query_id = result["query_id"]
        workbench.append_rows(
            "orders",
            ["id", "customer_id", "status", "amount", "created_at"],
            [(999993, 1, "paid", 1.0, "2024-12-31")],
        )
        self.assertTrue(workbench.session.conclusion(query_id)["is_stale"])
        self.assertEqual(workbench.change_events()[-1]["source_version_field"], "data_version")

        workbench.diagnose(SAMPLE_QUERY, timeout_ms=500, budget_ms=5000, threshold_ms=0, threshold_rows=0)
        self.assertFalse(workbench.session.conclusion(query_id)["is_stale"])
        workbench.refresh_statistics("customers")
        self.assertTrue(workbench.session.conclusion(query_id)["is_stale"])
        self.assertEqual(workbench.change_events()[-1]["source_version_field"], "stats_version")
        workbench.close()


if __name__ == "__main__":
    unittest.main()
