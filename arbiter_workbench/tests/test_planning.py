import unittest

from arbiter.planning import decide


class PlanningTest(unittest.TestCase):
    def test_small_change_prefers_incremental(self):
        d = decide(env_count=8, key_count=20, obs_total=800,
                   changed_env_count=1, dirty_key_count=2)
        self.assertEqual(d.selected, "incremental")
        self.assertFalse(d.within_tolerance)
        self.assertGreater(d.full_work, d.incremental_work)

    def test_tolerance_band_stable_tie_break(self):
        d = decide(env_count=4, key_count=5, obs_total=150,
                   changed_env_count=4, dirty_key_count=5)
        self.assertTrue(d.within_tolerance)
        self.assertEqual(d.selected, "incremental")
        self.assertIn("tolerance", d.tolerance_effect)
        self.assertEqual(d.not_triggered_boundaries["tolerance_ratio"], "1/10")

    def test_large_change_selects_full(self):
        d = decide(env_count=2, key_count=2, obs_total=10,
                   changed_env_count=2, dirty_key_count=2)
        self.assertEqual(d.selected, "full")
        self.assertFalse(d.within_tolerance)

    def test_deterministic_decision_fields(self):
        a = decide(6, 10, 300, 2, 4)
        b = decide(6, 10, 300, 2, 4)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
