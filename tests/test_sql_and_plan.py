from __future__ import annotations

import unittest
from pathlib import Path

from planner_app.errors import AppError
from planner_app.plan import apply_edit, compile_base, first_plan_difference
from planner_app.spec import load_workspace
from planner_app.sql_parser import parse_sql


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class SQLNormalisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = load_workspace(WORKSPACE)
        cls.orders = cls.workspace.table_map()["orders"]

    def plan_for(self, sql: str):
        parsed = parse_sql(sql)
        return compile_base(parsed, self.orders, 1.0, "normalisation")

    def test_whitespace_case_parentheses_and_in_order_are_normalised(self) -> None:
        first = self.plan_for(
            "SELECT status_code, COUNT(*) AS order_count, SUM(amount) AS total_amount, "
            "MIN(amount) AS min_amount, MAX(amount) AS max_amount "
            "FROM orders WHERE status_code IN (1, 2) AND (amount >= 10) "
            "GROUP BY status_code ORDER BY total_amount DESC"
        )
        second = self.plan_for(
            "select STATUS_CODE, count(*) as ORDER_COUNT, sum(AMOUNT) as total_amount, "
            "min(amount) AS min_amount, max(amount) as max_amount from ORDERS "
            "where ((status_code in (2,1))) and amount>=10 group by status_code "
            "order by TOTAL_AMOUNT desc;"
        )
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(first.summary, second.summary)

    def test_semantically_different_query_has_different_summary(self) -> None:
        first = self.plan_for(
            "SELECT status_code, COUNT(*) AS n FROM orders "
            "WHERE amount >= 10 GROUP BY status_code ORDER BY n DESC"
        )
        second = self.plan_for(
            "SELECT status_code, COUNT(*) AS n FROM orders "
            "WHERE amount >= 11 GROUP BY status_code ORDER BY n DESC"
        )
        self.assertNotEqual(first.summary, second.summary)
        self.assertNotEqual(first.to_bytes(), second.to_bytes())

    def test_invalid_syntax_reports_position(self) -> None:
        with self.assertRaises(AppError) as raised:
            parse_sql("SELECT FROM orders")
        self.assertEqual(raised.exception.code, "E_SYNTAX")
        self.assertIsNotNone(raised.exception.position)

    def test_first_plan_difference_locates_node(self) -> None:
        plan = self.plan_for("SELECT status_code, COUNT(*) AS n FROM orders GROUP BY status_code")
        changed, _effect = apply_edit(
            plan,
            {"type": "replace_scan_order", "target": "scan", "columns": ["order_day", "quantity", "status_code", "amount", "customer_id", "id"]},
            1,
        )
        difference = first_plan_difference(plan, changed)
        self.assertIsNotNone(difference)
        self.assertEqual(difference["node_id"], "scan")


if __name__ == "__main__":
    unittest.main()
