"""Vectorized evaluation, null/div-zero rules and determinism tests."""
import random
import unittest

from colval.executor import aggregate, evaluate_batch, merge_batches
from colval.executor import result_digest
from colval.expr import parse, normalize_text
from colval.formats import make_bitmap
from colval.plan import split_batches
from colval.quant import QuantContext, trunc_div
from colval.store import block_from_columns


def _block():
    n = 100
    v = [float((i % 9) - 4) for i in range(n)]  # includes 0.0 divisors
    return block_from_columns("blk-e", [
        ("a", "int64", make_bitmap([3, 17], n), [i % 13 for i in range(n)]),
        ("b", "float64", make_bitmap([5], n), v),
    ], 3, 1024)


def _run(block, expr, batch_rows, shuffle=False, seed=1):
    ctx = QuantContext(3)
    ast = parse(expr)
    batches = split_batches(block.row_count, batch_rows)
    if shuffle:
        random.Random(seed).shuffle(batches)
    results = [(b.row_start,
                evaluate_batch(block, ast, ctx, b.row_start, b.row_end))
               for b in batches]
    vec = merge_batches(results)
    return vec, result_digest(normalize_text(expr), 3, vec, "none", None)


class TestExecutor(unittest.TestCase):
    def test_determinism_across_batch_size_and_order(self):
        block = _block()
        expr = "(a + 2) * b - a / 3"
        digests = set()
        for batch_rows in (1, 7, 33, 100, 1000):
            for shuffle in (False, True):
                _, digest = _run(block, expr, batch_rows, shuffle=shuffle)
                digests.add(digest)
        self.assertEqual(len(digests), 1)

    def test_div_by_zero_flagged_and_nulled(self):
        block = _block()
        vec, _ = _run(block, "a / b", 1000)
        zero_rows = [i for i in range(100)
                     if i != 5 and float((i % 9) - 4) == 0.0]
        for i in zero_rows:
            self.assertTrue(vec.nulls[i])
            self.assertTrue(vec.div_zero[i])
        self.assertTrue(vec.nulls[3])   # null propagation
        self.assertTrue(vec.nulls[5])   # null divisor -> null, not div0 flag
        self.assertFalse(vec.div_zero[5])

    def test_ternary_and_logic(self):
        block = _block()
        vec, _ = _run(block, "if(a > 6 and b < 0, 1, 0)", 1000)
        # row 3: a null, b<0 true -> null AND true -> null -> result null
        self.assertTrue(vec.nulls[3])
        # row 17: a null, b<0 false -> null AND false -> false -> 0
        self.assertFalse(vec.nulls[17])
        self.assertEqual(int(vec.values[17]), 0)
        # row 5: b null, a>6 false -> false AND null -> false -> 0
        self.assertFalse(vec.nulls[5])
        self.assertEqual(int(vec.values[5]), 0)
        self.assertEqual(int(vec.values[0]), 0)

    def test_aggregations(self):
        block = _block()
        vec, _ = _run(block, "a * 2", 1000)
        valid = [int(v) for v, isn in zip(vec.values, vec.nulls) if not isn]
        self.assertEqual(aggregate(vec, "sum"), sum(valid))
        self.assertEqual(aggregate(vec, "count"), len(valid))
        self.assertEqual(aggregate(vec, "avg"),
                         trunc_div(sum(valid), len(valid)))
        self.assertIsNone(aggregate(vec, "none"))


if __name__ == "__main__":
    unittest.main()
