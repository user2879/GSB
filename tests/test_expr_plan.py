"""Expression normalization and batch split plan tests."""
import unittest

from colval.errors import PlanError
from colval.expr import expr_digest, normalize_text
from colval.plan import compile_plan, split_batches, validate_declared_batches


class TestExpr(unittest.TestCase):
    def test_whitespace_and_parens_same_digest(self):
        forms = [
            "a + b * c",
            "  a+b*c  ",
            "a + (b * c)",
            "(a) + ((b) * (c))",
            "((a + (b * c)))",
        ]
        digests = {expr_digest(f) for f in forms}
        self.assertEqual(len(digests), 1)

    def test_precedence_changes_digest(self):
        self.assertNotEqual(expr_digest("(a + b) * c"),
                            expr_digest("a + b * c"))

    def test_operators(self):
        self.assertEqual(normalize_text("x > 1 and y < 2 or not z"),
                         "(or (and (> x 1) (< y 2)) (not z))")
        self.assertEqual(normalize_text("a > 1 ? b : c"),
                         "(if (> a 1) b c)")
        self.assertEqual(normalize_text("if(a > 1, b, c)"),
                         "(if (> a 1) b c)")
        self.assertEqual(normalize_text("x && y || !z"),
                         "(or (and x y) (not z))")

    def test_syntax_error(self):
        with self.assertRaises(PlanError):
            normalize_text("a + * b")


class TestPlan(unittest.TestCase):
    def test_split_contiguous_gapless(self):
        batches = split_batches(250, 64)
        self.assertEqual([(b.row_start, b.row_end) for b in batches],
                         [(0, 64), (64, 128), (128, 192), (192, 250)])
        expect = 0
        for b in batches:
            self.assertEqual(b.row_start, expect)
            expect = b.row_end
        self.assertEqual(expect, 250)

    def test_declared_out_of_range_rejected(self):
        with self.assertRaises(PlanError) as cm:
            validate_declared_batches(100, [(0, 50), (50, 101)])
        self.assertEqual(cm.exception.code, "E_PLAN")

    def test_declared_overlap_or_gap_rejected(self):
        with self.assertRaises(PlanError):
            validate_declared_batches(100, [(0, 60), (50, 100)])
        with self.assertRaises(PlanError):
            validate_declared_batches(100, [(0, 40), (50, 100)])
        with self.assertRaises(PlanError):
            validate_declared_batches(100, [(0, 50)])

    def test_compile_plan(self):
        plan = compile_plan("qty * price", 250, 64)
        self.assertEqual(len(plan.batches), 4)
        self.assertEqual(plan.normalized, "(* qty price)")


if __name__ == "__main__":
    unittest.main()
