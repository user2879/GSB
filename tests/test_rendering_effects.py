from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.helpers import create_sample

import repair_studio  # noqa: F401
from repair_studio.rendering import Renderer, build_style_lut


class RenderingEffectTests(unittest.TestCase):
    def test_fill_is_deterministic_and_parameter_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "deterministic", width=72, height=56)
            base = sample.config.params
            first = Renderer(root / "cache_first")
            try:
                outcome_first = first.render(sample, base)
                outcome_first_again = first.render(sample, base)
            finally:
                first.close()
            second = Renderer(root / "cache_second")
            try:
                outcome_other_cache = second.render(sample, base)
            finally:
                second.close()

            self.assertEqual(outcome_first.final_png_sha256, outcome_first_again.final_png_sha256)
            self.assertEqual(outcome_first.final_png_sha256, outcome_other_cache.final_png_sha256)
            self.assertEqual(outcome_first.stage_summaries, outcome_other_cache.stage_summaries)
            self.assertEqual(outcome_first.stage_summaries["fill"]["fixed_bits"], 16)
            self.assertEqual(outcome_first.stage_summaries["fill"]["rounding"], "half_up")
            self.assertTrue(
                all(
                    stage in outcome_first_again.reused_stages
                    for stage in ["mask", "fill", "style", "composite"]
                )
            )

            radius_renderer = Renderer(root / "cache_radius")
            try:
                radius_two = radius_renderer.render(sample, replace(base, repair_radius=2))
                radius_four = radius_renderer.render(sample, replace(base, repair_radius=4))
                fewer_layers = radius_renderer.render(sample, replace(base, repair_iterations=4))
            finally:
                radius_renderer.close()
            self.assertNotEqual(
                radius_two.stage_summaries["fill"]["content_sha256"],
                radius_four.stage_summaries["fill"]["content_sha256"],
            )
            self.assertNotEqual(
                fewer_layers.stage_summaries["fill"]["content_sha256"],
                radius_two.stage_summaries["fill"]["content_sha256"],
            )

    def test_quantization_and_mask_policy_change_real_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "style_effects", width=64, height=48)
            base = sample.config.params
            renderer = Renderer(root / "cache")
            try:
                quant_eight = renderer.render(sample, replace(base, quant_levels=8, preserve_mask=False))
                quant_thirty_two = renderer.render(sample, replace(base, quant_levels=32, preserve_mask=False))
                preserved = renderer.render(sample, replace(base, quant_levels=8, preserve_mask=True))
            finally:
                renderer.close()

            self.assertNotEqual(
                quant_eight.stage_summaries["style"]["lut_sha256"],
                quant_thirty_two.stage_summaries["style"]["lut_sha256"],
            )
            self.assertNotEqual(quant_eight.final_png_sha256, quant_thirty_two.final_png_sha256)
            self.assertEqual(
                quant_eight.stage_summaries["composite"]["style_override_pixels"],
                base.mask.pixel_count(),
            )
            self.assertEqual(preserved.stage_summaries["composite"]["style_override_pixels"], 0)
            y_start = base.mask.y
            x_start = base.mask.x
            mask_area = (
                slice(y_start, y_start + base.mask.height),
                slice(x_start, x_start + base.mask.width),
            )
            self.assertFalse(
                (quant_eight.final_pixels[mask_area] == sample.source.pixels[mask_area]).all()
            )
            self.assertTrue(
                (preserved.final_pixels[mask_area] == sample.source.pixels[mask_area]).all()
            )

            lut_eight = build_style_lut(sample.style_stats, base.channel_weights_q, 8, 16)
            lut_thirty_two = build_style_lut(sample.style_stats, base.channel_weights_q, 32, 16)
            self.assertFalse((lut_eight == lut_thirty_two[:, :8]).all())


if __name__ == "__main__":
    unittest.main()
