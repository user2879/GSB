"""Fixed-point quantization, error bounds and stable tie-break tests."""
import unittest

from colval.expr import parse
from colval.formats import INT64_MAX, INT64_MIN
from colval.quant import QuantContext, batch_error_bound
from colval.quant import expression_error_bound, rank_candidates
from colval.quant import saturate, trunc_div


class TestQuant(unittest.TestCase):
    def test_trunc_toward_zero(self):
        self.assertEqual(trunc_div(7, 2), 3)
        self.assertEqual(trunc_div(-7, 2), -3)
        self.assertEqual(trunc_div(7, -2), -3)
        self.assertEqual(trunc_div(-7, -2), 3)

    def test_quantize_truncation_and_saturation(self):
        ctx = QuantContext(3)
        self.assertEqual(ctx.quantize(1.2345), 1234)
        self.assertEqual(ctx.quantize(-1.2345), -1234)
        self.assertEqual(ctx.quantize(0.0009), 0)
        self.assertEqual(ctx.quantize(1e30), INT64_MAX)
        self.assertEqual(ctx.quantize(-1e30), INT64_MIN)
        self.assertEqual(ctx.quantize_int(5), 5000)
        self.assertEqual(saturate(2 ** 70), INT64_MAX)

    def test_single_value_bound(self):
        ctx = QuantContext(3)
        self.assertAlmostEqual(ctx.single_value_bound, 0.001)
        for x in (0.1234, 3.9999, -7.0001):
            err = abs(x - ctx.dequantize(ctx.quantize(x)))
            self.assertLessEqual(err, ctx.single_value_bound)

    def test_expression_and_batch_bounds(self):
        ctx = QuantContext(3)
        stats = {"a": (10.0, 0.001), "b": (2.0, 0.001)}
        bound = expression_error_bound(parse("a * b + 1"), ctx, stats)
        self.assertAlmostEqual(bound, 0.002 + 0.01 + 0.000001 + 0.001)
        self.assertAlmostEqual(
            batch_error_bound(parse("a * b + 1"), ctx, stats, 64), bound * 64)

    def test_rank_candidates_tie_break(self):
        ctx = QuantContext(3)
        stats = {"a": (10.0, 0.001)}
        cands = [
            ("a * 2", parse("a * 2"), 20001),
            ("a + a", parse("a + a"), 20000),
            ("a * 3", parse("a * 3"), 30000),
        ]
        ranked = rank_candidates(cands, ctx, stats)
        self.assertEqual(ranked[0]["expr"], "a + a")
        self.assertTrue(ranked[0]["error_affected"])
        self.assertTrue(ranked[1]["error_affected"])
        self.assertFalse(ranked[2]["error_affected"])


if __name__ == "__main__":
    unittest.main()
