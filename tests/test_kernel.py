import unittest

from fxpipe.config import PipelineConfig
from fxpipe.kernel import (
    QuantContext,
    attention_weights,
    embed_lookup,
    isqrt,
    layer_norm,
    make_context,
    multihead_attention,
    segment_pool,
)


def config(gear="g1", frac_bits=16, heads=2, dim=8):
    return PipelineConfig(
        heads=heads, dim=dim, frac_bits=frac_bits, gear=gear,
        point_budget=1000, operator_quota=1000, max_workers=1, vocab_size=8,
    )


class KernelTests(unittest.TestCase):
    def test_isqrt(self):
        self.assertEqual(isqrt(0), 0)
        self.assertEqual(isqrt(1), 1)
        self.assertEqual(isqrt(15), 3)
        self.assertEqual(isqrt(16), 4)
        self.assertEqual(isqrt(99), 9)

    def test_gear_and_frac_bits_change_integers(self):
        row = [100, 200, -300]
        base = config(gear="g0")
        ctx_g0 = make_context(base, 32, 1000)
        ctx_g1 = make_context(config(gear="g1"), 32, 1000)
        ctx_g2 = make_context(config(gear="g2"), 32, 1000)
        out_g0 = embed_lookup(1, row, ctx_g0)
        out_g1 = embed_lookup(1, row, ctx_g1)
        out_g2 = embed_lookup(1, row, ctx_g2)
        self.assertNotEqual(out_g0, out_g1)
        self.assertNotEqual(out_g1, out_g2)

        ctx_f24 = make_context(config(gear="g1", frac_bits=24), 32, 1000)
        self.assertNotEqual(embed_lookup(1, row, ctx_g1), embed_lookup(1, row, ctx_f24))

    def test_all_integer_outputs(self):
        cfg = config()
        ctx = make_context(cfg, 16, 1250)
        rows = [[10, 20, 30, 40, 50, 60, 70, 80],
                [1, 2, 3, 4, 5, 6, 7, 8]]
        pooled = segment_pool(rows, ctx)
        self.assertEqual(pooled["count"], 2)
        self.assertTrue(all(isinstance(v, int) for v in pooled["sum"]))

        attended = multihead_attention("node", rows, 2, ctx)
        self.assertEqual(len(attended["vectors"]), 2)
        self.assertTrue(all(isinstance(v, int) for r in attended["vectors"] for v in r))

        normalized = layer_norm(rows, ctx)
        self.assertEqual(len(normalized["rows"]), 2)
        self.assertTrue(all(isinstance(v, int) for r in normalized["rows"] for v in r))

    def test_attention_weights_sum_to_1000_and_deterministic(self):
        weights_a = attention_weights("n", 0, 1, 5)
        weights_b = attention_weights("n", 0, 1, 5)
        self.assertEqual(weights_a, weights_b)
        self.assertEqual(sum(weights_a), 1000)
        self.assertEqual(len(weights_a), 5)

    def test_saturation_bounds(self):
        ctx = QuantContext(frac_bits=8, gear="g0", width=8, scale_permille=1000)
        huge = [10_000, -10_000]
        out = embed_lookup(0, huge, ctx)
        self.assertLessEqual(max(out), 127)
        self.assertGreaterEqual(min(out), -128)

    def test_pool_order_does_not_change_sum(self):
        cfg = config()
        ctx = make_context(cfg, 32, 1000)
        rows = [[1, 2], [3, 4], [5, 6]]
        forward = segment_pool(rows, ctx)
        reverse = segment_pool(list(reversed(rows)), ctx)
        # Saturating integer addition is commutative for this bounded range.
        self.assertEqual(forward["sum"], reverse["sum"])


if __name__ == "__main__":
    unittest.main()
