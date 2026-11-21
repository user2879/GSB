import unittest

from configbench.fixedpoint import (INT64_MAX, INT64_MIN, latency_stats,
                                    sat_add, sat_int64, sat_mul_fixed,
                                    trunc_div)


class FixedPointTest(unittest.TestCase):
    def test_trunc_toward_zero_negative(self):
        self.assertEqual(trunc_div(-7, 2), -3)
        self.assertEqual(trunc_div(7, 2), 3)
        self.assertEqual(trunc_div(-8, 2), -4)

    def test_saturate_bounds(self):
        self.assertEqual(sat_int64(INT64_MAX + 1), INT64_MAX)
        self.assertEqual(sat_int64(INT64_MIN - 1), INT64_MIN)
        self.assertEqual(sat_add(INT64_MAX, 1), INT64_MAX)
        self.assertEqual(sat_add(INT64_MIN, -1), INT64_MIN)

    def test_saturating_multiply(self):
        raw = sat_mul_fixed(INT64_MAX, 2, 0)
        self.assertEqual(raw, INT64_MAX)
        self.assertEqual(sat_mul_fixed(INT64_MIN, 2, 0), INT64_MIN)
        self.assertEqual(sat_mul_fixed(7, 3, 1), 2)  # 21/10 向零

    def test_latency_stats_deterministic_and_complete(self):
        vals = [10, 12, 9, 15, 11, 100, 13, 14, 8, 16]
        a = latency_stats(vals)
        b = latency_stats(list(reversed(vals)))
        self.assertEqual(a, b)  # 不依赖输入顺序
        self.assertEqual(a["scale"], 1000)
        self.assertEqual(a["truncation"], "toward_zero")
        self.assertEqual(a["count"], 10)  # 超预算样本不被剔除
        # 最近秩 P95: ceil(0.95*10)=10 -> 最大值 100
        self.assertEqual(a["p95"]["raw"], 100 * 1000)
        self.assertEqual(a["max"]["raw"], 100 * 1000)
        # 均值 208/10 = 20.8ms -> raw 20800
        self.assertEqual(a["mean"]["raw"], 20800)
        self.assertEqual(a["quant_error_bound_ms"]["raw"], 1)


if __name__ == "__main__":
    unittest.main()
