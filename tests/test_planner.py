import unittest

from colbench.planner import build_plan, parse_query
from colbench.sample_data import build_spec


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec("seg_a", 1)

    def test_projection_prunes_columns(self):
        plan = build_plan(
            "SELECT order_id FROM orders WHERE amount > 100", self.spec)
        self.assertEqual(plan.projected_columns, ["order_id"])
        self.assertIn("region", plan.pruned_columns)
        self.assertIn("priority", plan.pruned_columns)

    def test_predicate_kinds_pushdown(self):
        plan = build_plan(
            "SELECT order_id FROM orders WHERE amount = 50 "
            "AND region IN ('north', 'south') AND priority >= 1",
            self.spec)
        kinds = sorted(p.kind for p in plan.pushed)
        self.assertEqual(kinds, ["eq", "ge", "in"])
        self.assertEqual(plan.retained, [])

    def test_parentheses_and_whitespace_equivalent(self):
        a = build_plan(
            "SELECT order_id, region FROM orders WHERE amount >= 50 "
            "AND amount <= 150 AND priority = 2", self.spec)
        b = build_plan(
            "SELECT order_id, region FROM orders WHERE  ( ( amount >= 50 ) "
            "AND ( amount <= 150 ) )   AND priority = 2", self.spec)
        self.assertEqual(a.plan_digest, b.plan_digest)

    def test_unknown_column_rejected(self):
        from colbench.errors import SchemaError
        with self.assertRaises(SchemaError):
            build_plan("SELECT missing FROM orders", self.spec)

    def test_parse_in_and_ne(self):
        _, predicates = parse_query(
            "WHERE a IN (1, 2, 3) AND b <> 9")
        self.assertEqual(len(predicates), 2)
        kinds = {p.kind for p in predicates}
        self.assertEqual(kinds, {"in", "ne"})

    def test_pushdown_skips_blocks(self):
        plan = build_plan(
            "SELECT order_id FROM orders WHERE amount > 100000", self.spec)
        self.assertGreater(len(plan.skipped_blocks), 0)
        self.assertTrue(all(item["skip_reason"]
                            for item in plan.skipped_blocks))


if __name__ == "__main__":
    unittest.main()
