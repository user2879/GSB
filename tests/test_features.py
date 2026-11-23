"""业务要求 2：确定性特征抽取、补齐标注与定点量化记录。"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from retbench.config import build_config
from retbench.features import extract_text, extract_visual, projection_matrix
from retbench.fixedpoint import QuantSpec
from retbench.generator import default_config, generate_workspace
from retbench.samples import scan_samples


class FeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=5, inject_count=0)
        self.config = build_config(default_config())
        self.sample_set = scan_samples(
            f"{self.temp}/samples", self.config.resolution, self.config.vocabulary
        )
        self.sample = self.sample_set.samples["s000"]
        self.spec = QuantSpec(self.config.frac_bits, self.config.tier)

    def test_projection_matrix_is_deterministic(self) -> None:
        cfg_fp = "abc123"
        first = projection_matrix(8, 5, cfg_fp, "visual-full")
        second = projection_matrix(8, 5, cfg_fp, "visual-full")
        np.testing.assert_array_equal(first, second)
        other = projection_matrix(8, 5, "different", "visual-full")
        self.assertFalse(np.array_equal(first, other))

    def test_visual_extraction_elementwise_stable(self) -> None:
        first = extract_visual(self.sample, self.config, "cfgfp", self.spec, "full")
        second = extract_visual(self.sample, self.config, "cfgfp", self.spec, "full")
        np.testing.assert_array_equal(first.q(), second.q())
        self.assertEqual(len(first.tile_keys), self.config.grid[0] * self.config.grid[1])
        # 量化记录完整
        for key in ("step", "q", "residual", "tier", "frac_bits"):
            self.assertIn(key, first.quant)
        np.testing.assert_allclose(
            first.projected,
            first.q() * self.spec.step + first.quant["residual"],
            rtol=0, atol=1e-12,
        )

    def test_reduced_precision_drops_edge_stat(self) -> None:
        full = extract_visual(self.sample, self.config, "cfgfp", self.spec, "full")
        reduced = extract_visual(self.sample, self.config, "cfgfp", self.spec, "reduced")
        self.assertEqual(full.raw_stats.shape[1], 5)
        self.assertEqual(reduced.raw_stats.shape[1], 4)
        self.assertFalse(np.array_equal(full.q(), reduced.q()))

    def test_text_padding_positions_marked(self) -> None:
        target = self.sample.assertion_len() + 3
        text = extract_text(self.sample, self.config, "cfgfp", self.spec, target)
        self.assertEqual(text.projected.shape[0], target)
        self.assertEqual(text.padded_positions,
                         list(range(self.sample.assertion_len(), target)))
        self.assertTrue(all(text.assertion_refs[i]["padded"]
                            for i in text.padded_positions))

    def test_quant_tier_changes_integers(self) -> None:
        coarse = extract_visual(self.sample, self.config, "cfgfp",
                                QuantSpec(8, "coarse"), "full")
        fine = extract_visual(self.sample, self.config, "cfgfp",
                              QuantSpec(16, "fine"), "full")
        self.assertNotEqual(coarse.quant["step"], fine.quant["step"])
        self.assertFalse(np.array_equal(coarse.q(), fine.q()))


if __name__ == "__main__":
    unittest.main()
