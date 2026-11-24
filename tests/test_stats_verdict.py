import random
import unittest

from fppdiag.stats import (VERDICT_INDISTINGUISHABLE, VERDICT_OVER,
                           VERDICT_UNDER, compute_window_stats, decide,
                           evaluate_feature, tv_drift_bps)


class TestStats(unittest.TestCase):
    def test_group_order_independent(self):
        vals = [((i * 37) % 9000) - 4500 if i % 13 else None
                for i in range(200)]
        s1 = compute_window_stats(vals, 3, 4)
        shuffled = vals[:]
        random.Random(7).shuffle(shuffled)
        s2 = compute_window_stats(shuffled, 3, 4)
        self.assertEqual(s1.to_dict(), s2.to_dict())

    def test_mean_is_integer_and_scaled(self):
        s = compute_window_stats([100, 200, None], 2, 4)  # means 1.50 -> 15000
        self.assertEqual(s.mean_q, 15000)
        self.assertEqual(s.missing_bps, 3333)

    def test_decile_histogram_sums_to_count(self):
        vals = list(range(100))
        s = compute_window_stats(vals, 0, 2)
        self.assertEqual(sum(s.decile_hist), 100)

    def test_tv_bounds(self):
        from fppdiag.stats import reference_decile_edges
        ref_vals = list(range(100))
        edges = reference_decile_edges(ref_vals, 0, 2)
        uniform = compute_window_stats(ref_vals, 0, 2, edges)
        skewed = compute_window_stats([v * v for v in range(100)], 0, 2, edges)
        self.assertGreater(tv_drift_bps(uniform, skewed), 0)
        self.assertEqual(tv_drift_bps(uniform, uniform), 0)

    def test_verdict_three_classes(self):
        over = decide(drift_bps=5000, error_bps=10, threshold_bps=1500)
        under = decide(drift_bps=100, error_bps=10, threshold_bps=1500)
        ind = decide(drift_bps=1500, error_bps=200, threshold_bps=1500)
        self.assertEqual(over["verdict"], VERDICT_OVER)
        self.assertEqual(under["verdict"], VERDICT_UNDER)
        self.assertEqual(ind["verdict"], VERDICT_INDISTINGUISHABLE)
        self.assertTrue(ind["error_affected"])

    def test_stable_tie_break_never_over_on_equality(self):
        # drift - error == threshold exactly must not be "over"
        res = decide(drift_bps=1510, error_bps=10, threshold_bps=1500)
        self.assertNotEqual(res["verdict"], VERDICT_OVER)
        self.assertEqual(res["verdict"], VERDICT_INDISTINGUISHABLE)

    def test_evaluate_feature_structured(self):
        ref = compute_window_stats([1000, 1100, 1200], 3, 4)
        obs = compute_window_stats([5000, 5100, 5200], 3, 4)
        out = evaluate_feature("f", ref, obs, 1500, feature_bound_q=2,
                               uncertain_rows=0, scale_exp=4)
        self.assertIn(out["verdict"]["verdict"],
                      {VERDICT_OVER, VERDICT_INDISTINGUISHABLE})
        self.assertEqual(out["aggregation"].split(";")[0],
                         "sum_then_divide_half_away")


if __name__ == "__main__":
    unittest.main()
