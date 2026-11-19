"""Image-to-image branch: early denoise steps are replaced by image latent."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest

import numpy as np

from dbench.dag import IMAGE_ENCODE_STAGE, denoise_stage
from dbench.orchestrator import Orchestrator
from dbench.store import SnapshotStore
from tests.test_equivalence import _load, _run


class Img2ImgTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-i2i-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graph_contains_image_encoder_and_replaces_early_steps(self):
        cfg = _load("valid_img2img.json")
        report, arrays, _ = _run(cfg, self.tmp, "i2i")
        self.assertIn(IMAGE_ENCODE_STAGE, arrays)
        graph = report["dependency_graph"]
        for step in range(1, cfg.image_replacement_steps + 1):
            self.assertIn(IMAGE_ENCODE_STAGE,
                          graph[denoise_stage(step)]["deps"])
            self.assertTrue(graph[denoise_stage(step)]["replaced"])
        np.testing.assert_array_equal(
            arrays[denoise_stage(1)].shape, arrays[IMAGE_ENCODE_STAGE].shape
        )

    def test_routing_change_keeps_encoder_hit_and_recomputes_denoise(self):
        cfg = _load("valid_img2img.json")
        store = SnapshotStore(os.path.join(self.tmp, "i2i"))
        Orchestrator(cfg, store,
                     output_png=os.path.join(self.tmp, "a.png")).run()
        cfg2 = dataclasses.replace(cfg, image_replacement_steps=2)
        report2 = Orchestrator(
            cfg2, store, output_png=os.path.join(self.tmp, "b.png")
        ).run()
        self.assertIn(IMAGE_ENCODE_STAGE, report2["reuse"])
        self.assertIn("encode_prompt", report2["reuse"])
        self.assertIn(denoise_stage(4), report2["recompute"])


if __name__ == "__main__":
    unittest.main()
