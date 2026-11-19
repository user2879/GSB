from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import repair_studio  # noqa: F401
import numpy as np

from tests.helpers import create_sample

from repair_studio.rendering import Renderer


class ConcurrencyTests(unittest.TestCase):
    def test_single_and_multi_thread_outputs_and_failures_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "threaded", width=80, height=56)
            single_sample = replace(
                sample,
                config=replace(sample.config, thread_workers=1),
            )
            multi_sample = replace(
                sample,
                config=replace(sample.config, thread_workers=4),
            )
            single_renderer = Renderer(root / "single_cache")
            multi_renderer = Renderer(root / "multi_cache")
            try:
                single = single_renderer.render(single_sample, single_sample.config.params)
                multi = multi_renderer.render(multi_sample, multi_sample.config.params)
            finally:
                single_renderer.close()
                multi_renderer.close()

            self.assertTrue(np.array_equal(single.final_pixels, multi.final_pixels))
            self.assertEqual(single.final_png_sha256, multi.final_png_sha256)
            self.assertEqual(single.stage_summaries, multi.stage_summaries)
            self.assertLessEqual(multi.max_workers_observed, 4)
            self.assertGreaterEqual(multi.max_workers_observed, 1)
            self.assertEqual(single.stage_summaries["fill"]["failures"], [])
            self.assertEqual(multi.stage_summaries["fill"]["failures"], [])

            for stage in ("fill", "style", "composite"):
                task_keys = multi.stage_summaries[stage]["task_keys"]
                self.assertEqual(task_keys, sorted(task_keys))
                for block_row, key in enumerate(task_keys):
                    self.assertTrue(key.endswith(f"{block_row:08d}|{stage}"))


if __name__ == "__main__":
    unittest.main()
