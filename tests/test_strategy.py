"""业务要求 4：规模自适应策略选择真实改变计算路径。"""

from __future__ import annotations

import tempfile
import unittest

from retbench.config import build_config
from retbench.generator import default_config, generate_workspace
from retbench.samples import scan_samples
from retbench.strategy import select_strategy


class StrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=12, inject_count=0)
        self.config = build_config(default_config())
        self.sample_set = scan_samples(
            f"{self.temp}/samples", self.config.resolution, self.config.vocabulary
        )

    def test_strategy_switches_with_scale(self) -> None:
        names = self.sample_set.ordered_names()
        small = [self.sample_set.samples[n] for n in names[:4]]
        large = [self.sample_set.samples[n] for n in names[:10]]
        small_decision = select_strategy(self.config, small, 4)
        large_decision = select_strategy(self.config, large, 16)
        self.assertEqual(small_decision.strategy_id, "ST_FULL")
        self.assertEqual(large_decision.strategy_id, "ST_REDUCED")
        self.assertEqual(small_decision.feature_precision, "full")
        self.assertEqual(large_decision.feature_precision, "reduced")
        self.assertEqual(small_decision.tile_range, "full")
        self.assertEqual(large_decision.tile_range, "center_half")
        self.assertIn("hybrid", small_decision.channels)
        self.assertNotIn("hybrid", large_decision.channels)

    def test_decision_record_is_auditable_and_reproducible(self) -> None:
        names = self.sample_set.ordered_names()
        samples = [self.sample_set.samples[n] for n in names[:4]]
        first = select_strategy(self.config, samples, 4)
        second = select_strategy(self.config, samples, 4)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(all(row["strategy_id"] for row in first.scores))
        self.assertTrue(first.reason_codes)
        self.assertLessEqual(first.candidate_cap, self.config.candidate_cap)

    def test_cap_discount_reduces_candidate_limit(self) -> None:
        names = self.sample_set.ordered_names()
        large = [self.sample_set.samples[n] for n in names[:10]]
        decision = select_strategy(self.config, large, 16)
        expected = max(1, round(self.config.candidate_cap * 0.6))
        self.assertEqual(decision.candidate_cap, min(expected, len(large)))


if __name__ == "__main__":
    unittest.main()
