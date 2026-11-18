from __future__ import annotations

import unittest

from slowquery_workbench.errors import WorkbenchError
from slowquery_workbench.query import is_slow, merge_redundant_predicates, parse_select


class QueryParserTests(unittest.TestCase):
    def test_whitespace_and_case_produce_same_canonical_and_id(self) -> None:
        left = parse_select(
            "select * from orders O join customers C on O.customer_id=C.id "
            "where O.status='paid' and O.amount>=100 and C.city in ('Shanghai','Beijing') order by O.id"
        )
        right = parse_select(
            "  SELECT  *\nFROM orders AS o\nJOIN customers AS c ON o.customer_id = c.id\n"
            "WHERE o.status = 'paid' AND o.amount >= 100 AND c.city IN ( 'Shanghai' , 'Beijing' )\n"
            "ORDER BY o.id ASC ; "
        )
        self.assertEqual(left.canonical_sql, right.canonical_sql)
        self.assertEqual(left.query_id, right.query_id)
        self.assertEqual(left.table_names, ("customers", "orders"))

    def test_supported_ranges_in_and_join(self) -> None:
        query = parse_select(
            "SELECT id, amount FROM orders o JOIN customers c ON o.customer_id = c.id "
            "WHERE amount BETWEEN 100 AND 200 AND status IN ('paid','shipped') AND o.id >= 3"
        )
        self.assertEqual(len(query.predicates), 3)
        self.assertEqual(query.predicates[0].kind, "between")
        self.assertEqual(query.predicates[1].kind, "in")
        self.assertIn("BETWEEN 100 AND 200", query.canonical_sql)

    def test_predicate_merge_removes_weaker_lower_bound(self) -> None:
        query = parse_select(
            "SELECT id FROM orders WHERE amount >= 10 AND amount >= 100 AND status = 'paid'"
        )
        merged = merge_redundant_predicates(query.predicates)
        self.assertIsNotNone(merged)
        predicates, rationale = merged
        rendered = [item.sql() for item in predicates]
        self.assertIn("amount >= 100", rendered)
        self.assertNotIn("amount >= 10", rendered)
        self.assertTrue(rationale)

    def test_rejects_subquery_aggregate_and_dml(self) -> None:
        unsupported = [
            "SELECT * FROM orders WHERE id IN (SELECT id FROM orders)",
            "SELECT count(*) FROM orders",
            "DELETE FROM orders WHERE id = 1",
            "UPDATE orders SET status='x'",
            "INSERT INTO orders(id) VALUES(1)",
            "SELECT * FROM orders WHERE status='paid' OR status='pending'",
            "SELECT * FROM orders o JOIN customers c ON o.customer_id=c.id JOIN customers c2 ON c2.id=c.id",
        ]
        for sql in unsupported:
            with self.subTest(sql=sql):
                with self.assertRaises(WorkbenchError) as captured:
                    parse_select(sql)
                self.assertEqual(captured.exception.code, "E_PARSE")

    def test_slow_threshold_uses_elapsed_or_scan_rows(self) -> None:
        self.assertTrue(is_slow(10, 1, 10, 1000))
        self.assertTrue(is_slow(1, 1000, 10, 1000))
        self.assertFalse(is_slow(9, 999, 10, 1000))
        with self.assertRaises(WorkbenchError) as captured:
            is_slow(1, 1, -1, 1)
        self.assertEqual(captured.exception.code, "E_SESSION")


if __name__ == "__main__":
    unittest.main()
