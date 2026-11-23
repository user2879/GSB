import unittest

from arbworkbench.spec import load_specification
from arbworkbench.engine import ArbitrationEngine
from arbworkbench.statistics_view import distribution_stats

from tests import SAMPLE_ENV, SAMPLE_VIEWS


class StatisticsViewTests(unittest.TestCase):
    def test_distribution_stats_are_numpy_derived(self):
        spec = load_specification(SAMPLE_ENV, SAMPLE_VIEWS)
        engine = ArbitrationEngine(spec)
        kc = engine.arbitrate_key(spec.views["view-production"], "timeout_ms")
        stats = distribution_stats(kc)
        self.assertGreaterEqual(stats["sample_count"], 3)
        self.assertIn("mean_q", stats)
        self.assertGreaterEqual(stats["spread_q"], 0)
        self.assertIsInstance(stats["median_q"], float)


if __name__ == "__main__":
    unittest.main()
