import unittest

from coldstart_diag.analytics import cold_ratio_heat_hist, queue_pressure_series, wait_percentiles
from coldstart_diag.samples import base_manifest
from coldstart_diag.engine import Engine


class AnalyticsTest(unittest.TestCase):
    def setUp(self):
        m = base_manifest(1, 1, 10000, 2500)
        self.payloads = [r.to_dict() for r in Engine(m, "fn-order-processor", 1).run_all(m["call_sequence"])]

    def test_percentiles_keys(self):
        pct = wait_percentiles(self.payloads)
        self.assertEqual(set(pct), {"p50", "p90", "p95", "p99"})

    def test_heat_hist_sums_to_minutes(self):
        hist = cold_ratio_heat_hist(self.payloads, 10000)
        self.assertEqual(sum(hist.values()), len(self.payloads))

    def test_queue_series_deterministic(self):
        a = queue_pressure_series(self.payloads)
        b = queue_pressure_series(self.payloads)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(self.payloads))


if __name__ == "__main__":
    unittest.main()
