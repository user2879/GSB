import unittest

from graphdiag.core.errors import ScaleError
from graphdiag.core.fixedpoint import (
    FP_MAX,
    Quantizer,
    build_path_terms,
    saturating_add,
)


class FixedPointTest(unittest.TestCase):
    def test_truncation_direction(self):
        q = Quantizer(0)  # 1 ulp == 1_000_000 micro
        self.assertEqual(q.quantize(1_999_999), 1)
        self.assertEqual(q.quantize(999_999), 0)
        q3 = Quantizer(3)  # 1 ulp == 1_000 micro
        self.assertEqual(q3.quantize(1_234_567), 1234)
        self.assertEqual(q3.micro_per_ulp, 1000)

    def test_saturating_add(self):
        total, saturated = saturating_add(FP_MAX - 5, 10)
        self.assertTrue(saturated)
        self.assertEqual(total, FP_MAX)
        total, saturated = saturating_add(3, 4)
        self.assertFalse(saturated)
        self.assertEqual(total, 7)

    def test_scale_domain_rejection(self):
        q = Quantizer(6)
        huge = (FP_MAX // 10 ** 6 + 10) * 1_000_000
        with self.assertRaises(ScaleError) as ctx:
            q.check_domain(huge, huge)
        self.assertEqual(ctx.exception.code, "E_SCALE")

    def test_scale_domain_accepts_normal(self):
        Quantizer(6).check_domain(10_000_000, 50_000_000)  # must not raise

    def test_path_error_terms_and_bounds(self):
        q = Quantizer(0)
        edges = [(0, 1, 1_500_000), (1, 2, 700_000), (2, 3, 1_900_000)]
        terms = build_path_terms(edges, q)
        self.assertEqual([t.running_edges for t in terms], [1, 2, 3])
        self.assertEqual([t.conservative_bound_ulp for t in terms], [1, 2, 3])
        self.assertEqual([t.edge_remainder_micro for t in terms],
                         [500_000, 700_000, 900_000])
        self.assertEqual(terms[-1].exact_bound_micro, 2_100_000)
        self.assertEqual(terms[-1].conservative_bound_micro, 3_000_000)

    def test_quantizer_rejects_bad_exponent(self):
        with self.assertRaises(ScaleError):
            Quantizer(9)


if __name__ == "__main__":
    unittest.main()
