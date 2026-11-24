import unittest

from fppdiag.spec import StageSpec
from fppdiag.stages import run_bucket
from fppdiag.stats import (VERDICT_INDISTINGUISHABLE, VERDICT_OVER,
                           VERDICT_UNDER, decide)


class TestIndistinguishable(unittest.TestCase):
    def test_error_interval_straddles_threshold(self):
        # drift 1500, error 100, threshold 1500 -> interval [1400,1600]
        res = decide(1500, 100, 1500)
        self.assertEqual(res["verdict"], VERDICT_INDISTINGUISHABLE)
        self.assertTrue(res["error_affected"])
        self.assertLessEqual(res["interval_lower_bps"], 1500)
        self.assertGreaterEqual(res["interval_upper_bps"], 1500)

    def test_clearly_over_and_clearly_under(self):
        self.assertEqual(decide(2000, 50, 1500)["verdict"], VERDICT_OVER)
        self.assertEqual(decide(1000, 50, 1500)["verdict"], VERDICT_UNDER)

    def test_bucket_boundary_uncertain_flag(self):
        stage = StageSpec("b", "bucket", 0, None, ("up",),
                          {"edges": [10, 20]})
        # value 11 is within 1 ulp of edge 10 -> boundary uncertain
        res = run_bucket(stage, [11, 15, 25, None], in_exp=0, input_bound_ulp=1)
        self.assertTrue(res.boundary_uncertain[0])
        self.assertFalse(res.boundary_uncertain[1])
        self.assertFalse(res.boundary_uncertain[2])
        self.assertIsNone(res.values[3])
        # x=11 -> bucket 1 (edges 10,20), center (10+20)/2 = 15;
        # x=15 -> bucket 1 center 15; x=25 -> outer bucket collapsed to edge 20.
        self.assertEqual(res.values[0], 15)
        self.assertEqual(res.values[1], 15)
        self.assertEqual(res.values[2], 20)

    def test_boundary_uncertain_grows_with_bound(self):
        stage = StageSpec("b", "bucket", 0, None, ("up",),
                          {"edges": [10, 20]})
        tight = run_bucket(stage, [12], 0, input_bound_ulp=1)
        loose = run_bucket(stage, [12], 0, input_bound_ulp=5)
        self.assertFalse(tight.boundary_uncertain[0])
        self.assertTrue(loose.boundary_uncertain[0])


if __name__ == "__main__":
    unittest.main()
