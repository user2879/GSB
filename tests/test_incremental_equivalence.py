from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import repair_studio  # noqa: F401
import numpy as np

from tests.helpers import create_sample

from repair_studio.config import apply_command, command_affected_stages
from repair_studio.engine import render_full_reference
from repair_studio.rendering import Renderer


class IncrementalEquivalenceTests(unittest.TestCase):
    def test_every_command_matches_fresh_full_render(self) -> None:
        commands = [
            {"op": "set_repair_radius", "value": 3},
            {"op": "set_style_weights", "weights": [0.2, 0.7, 0.5]},
            {"op": "set_quant_levels", "value": 32},
            {"op": "set_mask_rect", "rect": {"x": 12, "y": 10, "width": 30, "height": 22}},
            {"op": "set_preserve_mask", "value": True},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "incremental", width=64, height=48, commands=commands)
            incremental = Renderer(root / "incremental_cache")
            params = sample.config.params
            baseline = incremental.render(sample, params, command_index=0)
            self.assertTrue(baseline.final_pixels.any())
            try:
                for index, command in enumerate(commands, start=1):
                    previous = params
                    params = apply_command(params, sample.config.width, sample.config.height, command)
                    expected = command_affected_stages(str(command["op"]))
                    outcome = incremental.render(
                        sample,
                        params,
                        command_index=index,
                        expected_invalidated_stages=expected,
                    )
                    self.assertEqual(outcome.invalidated_stages, expected)
                    self.assertEqual(
                        sorted(outcome.invalidated_stages + outcome.reused_stages),
                        ["composite", "fill", "mask", "style"],
                    )
                    reference = render_full_reference(
                        sample,
                        root / f"reference_cache_{index}",
                        params,
                    )
                    self.assertTrue(
                        np.array_equal(outcome.final_pixels, reference.final_pixels),
                        msg=f"pixel mismatch after command {index}",
                    )
                    self.assertEqual(outcome.final_png_sha256, reference.final_png_sha256)
                    self.assertEqual(outcome.stage_summaries, reference.stage_summaries)
                    self.assertEqual(outcome.params_fingerprint, params.fingerprint())
                    self.assertNotEqual(previous.fingerprint(), params.fingerprint())
            finally:
                incremental.close()

    def test_reused_stage_does_not_execute_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(
                root,
                "reuse",
                width=56,
                height=40,
                commands=[{"op": "set_style_weights", "weights": [0.1, 0.2, 0.3]}],
            )
            renderer = Renderer(root / "cache")
            calls = {"fill": 0, "style": 0}
            original_fill = renderer._execute_fill
            original_style = renderer._execute_style

            def count_fill(*args: object, **kwargs: object) -> object:
                calls["fill"] += 1
                return original_fill(*args, **kwargs)

            def count_style(*args: object, **kwargs: object) -> object:
                calls["style"] += 1
                return original_style(*args, **kwargs)

            renderer._execute_fill = count_fill  # type: ignore[method-assign]
            renderer._execute_style = count_style  # type: ignore[method-assign]
            try:
                baseline = renderer.render(sample, sample.config.params)
                self.assertEqual(calls, {"fill": 1, "style": 1})
                changed = apply_command(
                    sample.config.params,
                    sample.config.width,
                    sample.config.height,
                    {"op": "set_style_weights", "weights": [0.2, 0.8, 0.4]},
                )
                outcome = renderer.render(sample, changed)
                self.assertIn("fill", outcome.reused_stages)
                self.assertNotIn("fill", outcome.invalidated_stages)
                self.assertEqual(calls, {"fill": 1, "style": 2})
                self.assertNotEqual(
                    baseline.stage_summaries["style"]["cache_key"],
                    outcome.stage_summaries["style"]["cache_key"],
                )
            finally:
                renderer._execute_fill = original_fill  # type: ignore[method-assign]
                renderer._execute_style = original_style  # type: ignore[method-assign]
                renderer.close()


if __name__ == "__main__":
    unittest.main()
