import tempfile
import unittest
from pathlib import Path

from mm_pipeline.config import parse_config
from mm_pipeline.fusion import fuse_region, _quantized_equal
from mm_pipeline.models import RegionFeature

from .fixtures import build_corpus


def _feature(sample_id: str, region: int, visual: list[int], text: list[int],
             key: str = "k") -> RegionFeature:
    return RegionFeature(
        sample_id=sample_id,
        region=region,
        visual_vector=visual,
        text_vector=text,
        backfilled_assertions=[],
        referenced_assertions=[],
        referenced_regions=[region],
        cache_key=key,
    )


class FusionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mm-fuse-"))
        _, config_path = build_corpus(self.tmp, count=1)
        self.config = parse_config(config_path)
        self.samples_root, _ = build_corpus(self.tmp, count=1)

    def _sample(self):
        from mm_pipeline.loader import load_samples
        return load_samples(self.samples_root, self.config)[0]

    def test_modality_tie_after_quantization_is_neutral_and_flagged(self):
        sample = self._sample()
        s = 1 << self.config.frac_bits
        vec_len_v = self.config.visual_dim
        vec_len_t = self.config.text_dim
        # Equal per-dimension magnitude so weighted modality scores tie.
        visual = [s // vec_len_v] * vec_len_v
        text = [s // vec_len_t] * vec_len_t
        feature = _feature(sample.sample_id, 0, visual, text)
        result = fuse_region(feature, sample, self.config)
        self.assertTrue(result.boundary)
        self.assertIn("modality_tie", result.boundary_reasons)
        self.assertEqual(result.category, "neutral")
        self.assertEqual(result.visual_quantized, result.text_quantized)

    def test_quantized_equality_depends_on_level(self):
        # Two weighted scores equal after a coarse 4-bit shift but distinct
        # after a fine 2-bit shift (difference of 5).
        # Force coarse level 16 (shift 4): differences under 16 collapse.
        from mm_pipeline.config import PipelineConfig
        coarse_cfg = PipelineConfig(**{**self.config.to_dict(), "quant_level": 16,
                                       "frac_bits": 24})
        fine_cfg = PipelineConfig(**{**self.config.to_dict(), "quant_level": 8,
                                     "frac_bits": 24})
        self.assertTrue(_quantized_equal(1000, 1005, coarse_cfg))
        self.assertFalse(_quantized_equal(1000, 1005, fine_cfg))

    def test_contribution_order_is_deterministic(self):
        sample = self._sample()
        s = 1 << self.config.frac_bits
        feature = _feature(
            sample.sample_id, 1,
            [100] * self.config.visual_dim,
            [100] * self.config.text_dim,
        )
        r1 = fuse_region(feature, sample, self.config)
        r2 = fuse_region(feature, sample, self.config)
        order1 = [(c.modality, c.relation_type) for c in r1.contributions]
        order2 = [(c.modality, c.relation_type) for c in r2.contributions]
        self.assertEqual(order1, order2)
        # Visual headline contribution always precedes textual ones.
        self.assertEqual(order1[0][0], "visual")


if __name__ == "__main__":
    unittest.main()
