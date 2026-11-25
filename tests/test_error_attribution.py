import unittest

from arbplatform.engine import (
    CAUSE_ERROR_INDISTINGUISHABLE,
    CAUSE_REJECTED_RATIO_HIGH,
    ArbitrationEngine,
)
from arbplatform.fixedpoint import FixedPoint
from tests.test_engine import manifest, obs


class ErrorAttributionTest(unittest.TestCase):
    def test_error_indistinguishable_when_within_bound(self):
        # tolerance 5 raw units at scale 2 -> bound = 2 ulp + 500 = 502 ulp.
        m = manifest("weighted_majority", tolerance=5, scale=2)
        engine = ArbitrationEngine(m)
        data = [
            obs("e1", "i1", 1, 10),
            obs("e2", "i2", 1, 11),  # diff 100 ulp, inside the 502 ulp bound
            obs("e3", "i3", 1, 10),
        ]
        verdict = engine.arbitrate_view(m.view("v"), 1, data).verdicts["k"]
        self.assertTrue(verdict.error_influenced)
        self.assertIn(CAUSE_ERROR_INDISTINGUISHABLE, verdict.causes)
        self.assertEqual(verdict.error_bound_ulp, 502)

    def test_rejected_ratio_high_attribution(self):
        m = manifest("latest_revision", tolerance=0)
        engine = ArbitrationEngine(m)
        data = [
            obs("e1", "i1", 3, 10),
            obs("e2", "i2", 1, 20),
            obs("e3", "i3", 1, 30),
        ]
        verdict = engine.arbitrate_view(m.view("v"), 1, data).verdicts["k"]
        # Two of three environments rejected -> ratio 0.666 > 0.5 threshold.
        self.assertIn(CAUSE_REJECTED_RATIO_HIGH, verdict.causes)
        self.assertEqual(len(verdict.rejected), 2)

    def test_stable_tie_break_prefers_smaller_quantized(self):
        fp = FixedPoint(2)
        m = manifest("latest_revision", tolerance=0)
        engine = ArbitrationEngine(m)
        data = [
            obs("e1", "i1", 2, 40),
            obs("e2", "i2", 2, 40),
            obs("e3", "i3", 2, 40),
        ]
        verdict = engine.arbitrate_view(m.view("v"), 1, data).verdicts["k"]
        self.assertEqual(verdict.authority_q, fp.quantize(40).q)
        self.assertEqual(verdict.adopted, ("e1", "e2", "e3"))


if __name__ == "__main__":
    unittest.main()
