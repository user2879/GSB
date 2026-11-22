"""Cross-check the integer fixed-point SLI math against NumPy floats."""

import unittest

from meshbench.budget import SCALE, WindowCounters, compute_receipt
from meshbench.config import load_config

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class NumpyCrossCheckTests(unittest.TestCase):
    def setUp(self):
        self.route = load_config(os.path.join(ROOT, "samples", "rules_v1.json")).routes["orders-api"]

    def test_fixed_point_matches_numpy_float_within_quant_error(self):
        import numpy as np
        rng = np.random.default_rng(seed=42)
        for trial in range(25):
            success, failure, timeout, slow = (int(x) for x in rng.integers(0, 40, size=4))
            counters = WindowCounters(success, failure, timeout, slow)
            if counters.total == 0:
                continue
            payload = compute_receipt(self.route, 1, trial, counters).payload
            expected_avail = (success + slow) / counters.total
            actual_avail = payload["availability_sli_scaled"] / SCALE
            self.assertLessEqual(abs(actual_avail - expected_avail), 1.0 / SCALE + 1e-12)
            expected_bad = (failure + timeout + slow) / counters.total
            actual_bad = payload["bad_rate_scaled"] / SCALE
            self.assertLessEqual(abs(actual_bad - expected_bad), 1.0 / SCALE + 1e-12)

    def test_numpy_percentile_used_by_view(self):
        import numpy as np
        from meshbench.view import _numpy_percentile
        values = list(range(1, 101))
        self.assertEqual(_numpy_percentile(values, 95),
                         int(round(float(np.percentile(np.array(values), 95)))))


if __name__ == "__main__":
    unittest.main()
