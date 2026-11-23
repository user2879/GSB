"""业务要求 3：对齐打分、多通道融合、同分裁决与定点语义。"""

from __future__ import annotations

import tempfile
import unittest

from retbench.alignment import (
    CHANNEL_ORDER,
    align_pair,
    fuse_candidate,
    lexical_score,
    run_query,
    select_tile_keys,
)
from retbench.config import build_config
from retbench.features import extract_text, extract_visual, stable_normalize
from retbench.fixedpoint import QuantSpec
from retbench.generator import default_config, generate_workspace
from retbench.samples import scan_samples
from retbench.strategy import select_strategy

import numpy as np


class AlignmentRecallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=6, inject_count=0)
        self.config = build_config(default_config())
        self.sample_set = scan_samples(
            f"{self.temp}/samples", self.config.resolution, self.config.vocabulary
        )
        self.names = self.sample_set.ordered_names()
        self.spec = QuantSpec(self.config.frac_bits, self.config.tier)
        self.visual = {
            name: extract_visual(self.sample_set.samples[name], self.config,
                                 "cfgfp", self.spec, "full")
            for name in self.names
        }
        target = max(s.assertion_len() for s in self.sample_set.samples.values())
        self.text = {
            name: extract_text(self.sample_set.samples[name], self.config,
                               "cfgfp", self.spec, target)
            for name in self.names
        }

    def test_stable_normalization_sums_to_one(self) -> None:
        vec = np.array([1000.0, 999.0, -5.0])
        out = stable_normalize(vec)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=12)
        big = stable_normalize(np.array([1e9, 1e9 - 50]))
        self.assertTrue(np.isfinite(big).all())

    def test_tile_range_center_half_subset(self) -> None:
        full = select_tile_keys(self.config.grid, "full")
        center = select_tile_keys(self.config.grid, "center_half")
        self.assertEqual(full, sorted(full))
        self.assertTrue(set(center).issubset(set(full)))
        self.assertLess(len(center), len(full))

    def test_alignment_uses_fixed_tile_order(self) -> None:
        result = align_pair(self.text[self.names[0]], self.visual[self.names[1]],
                            self.config, self.spec, "full")
        tiles = [tuple(item["tile"]) for item in result.tile_similarities]
        self.assertEqual(tiles, sorted(tiles))
        self.assertIsInstance(result.alignment_q, int)

    def test_channel_contribution_decomposition_complete(self) -> None:
        decision = select_strategy(self.config,
                                   [self.sample_set.samples[n] for n in self.names],
                                   self.config.active_tier)
        query = run_query(self.names[0], self.visual, self.text,
                          self.sample_set.samples, self.config, self.spec,
                          decision, self.names)
        payload = query.to_dict()
        for candidate in payload["candidates"]:
            self.assertEqual(set(candidate["channel_contrib"]), set(CHANNEL_ORDER))
            self.assertEqual(
                candidate["fused_q"],
                sum(candidate["channel_contrib"].values()),
            )

    def test_tie_break_by_sample_name(self) -> None:
        decision = select_strategy(self.config,
                                   [self.sample_set.samples[n] for n in self.names],
                                   self.config.active_tier)
        query = run_query(self.names[0], self.visual, self.text,
                          self.sample_set.samples, self.config, self.spec,
                          decision, self.names)
        # 手工构造完全同分的两个候选，验证样本名裁决
        first = query.candidates[0]
        cloned = type(first)(**{**first.__dict__})
        cloned.sample = "zzzz"
        ordered = sorted([first, cloned], key=lambda item: item.tie_break_key())
        self.assertEqual(ordered[0].sample, first.sample)

    def test_temperature_changes_fused_integer(self) -> None:
        alignment = align_pair(self.text[self.names[0]], self.visual[self.names[0]],
                               self.config, self.spec, "full")
        lex = lexical_score(self.text[self.names[0]],
                            self.sample_set.samples[self.names[0]], self.spec)
        cold = fuse_candidate(alignment, lex, self.config, self.spec,
                              list(CHANNEL_ORDER))
        warm_config = build_config(default_config())
        warm_config.fusion_temperature = 4.0
        hot = fuse_candidate(alignment, lex, warm_config, self.spec,
                             list(CHANNEL_ORDER))
        self.assertNotEqual(cold.fused_q, hot.fused_q)


if __name__ == "__main__":
    unittest.main()
