import unittest

from coldstart_diag.errors import SpecError
from coldstart_diag.samples import base_manifest
from coldstart_diag.versioning import assert_monotonic, compare_version_stats, diff_views


class VersioningTest(unittest.TestCase):
    def test_monotonic_guard(self):
        m1, m2 = base_manifest(1, 1, 10000, 2500), base_manifest(2, 2, 10000, 4000)
        assert_monotonic(m1, m2)
        with self.assertRaises(SpecError):
            assert_monotonic(m2, m1)

    def test_diff_stable(self):
        old = [{"minute": 0, "calls": 10, "cold_ratio_q": 100, "error_block": {"x": 1}}]
        new = [{"minute": 0, "calls": 12, "cold_ratio_q": 100, "error_block": {"x": 1}}]
        changes = diff_views(old, new)
        self.assertEqual(changes[0]["minute"], 0)
        self.assertEqual(changes[0]["field"], "calls")
        self.assertEqual(changes[0]["old"], 10)

    def test_indistinguishable_inside_bound(self):
        a = {"overall_cold_ratio_q": 100, "overall_wait_ratio_q": 50,
             "avg_occupancy_q": 80, "agg_error_bound_ulp": 5}
        b = dict(a, overall_cold_ratio_q=104)
        verdicts = compare_version_stats(a, b)
        cold = next(v for v in verdicts if v["field"] == "overall_cold_ratio_q")
        self.assertEqual(cold["verdict"], "INDISTINGUISHABLE")
        c = dict(a, overall_cold_ratio_q=200)
        cold2 = next(v for v in compare_version_stats(a, c)
                     if v["field"] == "overall_cold_ratio_q")
        self.assertEqual(cold2["verdict"], "DISTINGUISHABLE")


if __name__ == "__main__":
    unittest.main()
