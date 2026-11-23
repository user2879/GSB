import unittest

from alert_pipeline import fixedpoint as fp
from alert_pipeline.correlation import Correlation
from alert_pipeline.models import RootCandidate
from tests.test_dedup_correlation import alert, rules
from alert_pipeline.dedup import aggregate_alerts


def shim_aggregates():
    spec = rules()
    alerts = [alert(1, 0, fp="fp_000000000001"),
              alert(2, 10, fp="fp_000000000001"),
              alert(3, 5000, fp="fp_000000000002")]
    return aggregate_alerts(alerts, spec)


class FixedPointTest(unittest.TestCase):
    def test_floor_fixed_and_three_segment_bounds(self):
        aggs = shim_aggregates()
        stats = fp.suppression_ratios(aggs, scale_k=4)
        global_stat = next(s for s in stats if s.key == "__global__")
        # 1 suppressed / 3 total = 0.3333 floor at S=1e4.
        self.assertEqual(global_stat.fixed_value, 3333)
        self.assertEqual(global_stat.decimal(), "0.3333")
        self.assertEqual(global_stat.truncation, "floor")
        breakdown = global_stat.ulp_breakdown
        self.assertEqual(set(breakdown), {
            fp.QUANT, fp.WINDOW, fp.LAYER})
        self.assertEqual(global_stat.dominant_segment, fp.WINDOW)

    def test_hit_rate_and_average_use_integer_arithmetic(self):
        aggs = shim_aggregates()
        corrs = [
            Correlation(aggs[0].aggregate_id, 1,
                        (RootCandidate("x", "upstream", 1, ("a", "x")),)),
            Correlation(aggs[1].aggregate_id, 1, ())]
        hit = fp.root_cause_hit_rate(aggs, corrs, 4)
        self.assertEqual(hit.fixed_value, 5000)  # 1/2 floor
        self.assertEqual(hit.ulp_breakdown[fp.LAYER], 5000)
        avg = fp.average_aggregate_count(aggs, service_count=4, scale_k=4)
        # 2 aggregates / 4 services = 0.5
        self.assertEqual(avg.fixed_value, 5000)

    def test_indistinguishable_tie_break_is_stable_and_flagged(self):
        aggs = shim_aggregates()
        stats = fp.suppression_ratios(aggs, scale_k=4)
        ranked = fp.rank_fingerprints(stats)
        keys = [s.key for s in ranked]
        self.assertEqual(keys, sorted(keys, key=lambda k: next(
            -s.fixed_value for s in ranked if s.key == k)))
        # Both fingerprints sit inside the synthetic ULP bound -> flagged.
        self.assertTrue(all(s.error_indistinguishable for s in ranked))


if __name__ == "__main__":
    unittest.main()
