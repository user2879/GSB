"""加载层校验：PNG 截断、相机缺失、分辨率/内参/视场角异常隔离。"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np
from PIL import Image

from recon_workbench.errors import BizCode
from recon_workbench.loader import load_sample
from recon_workbench.utils import atomic_write_json, read_json

from ._helpers import SampleHarness


class LoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_clean_sample_loads(self) -> None:
        result = load_sample(self.harness.sample_dir)
        self.assertEqual(len(result.views), 5)
        self.assertEqual(result.failures, [])
        view = result.views[0]
        self.assertEqual(view.gray.ndim, 2)
        self.assertEqual(view.color.ndim, 3)

    def test_truncated_png_isolated(self) -> None:
        path = os.path.join(self.harness.sample_dir, "view_02.png")
        with open(path, "r+b") as handle:
            handle.truncate(20)
        result = load_sample(self.harness.sample_dir)
        codes = {f.code for f in result.failures}
        self.assertIn(BizCode.E_PNG_TRUNCATED, codes)
        self.assertEqual({f.view for f in result.failures}, {2})
        self.assertEqual(len(result.views), 4)

    def test_missing_camera_isolated(self) -> None:
        os.remove(os.path.join(self.harness.sample_dir,
                               "view_01.camera.json"))
        result = load_sample(self.harness.sample_dir)
        self.assertIn(BizCode.E_CAMERA_MISSING,
                      {f.code for f in result.failures})
        self.assertEqual(len(result.views), 4)

    def test_singular_intrinsic_rejected(self) -> None:
        path = os.path.join(self.harness.sample_dir,
                            "view_03.camera.json")
        camera = read_json(path)
        camera["intrinsic"] = [[0.0] * 3 for _ in range(3)]
        atomic_write_json(path, camera)
        result = load_sample(self.harness.sample_dir)
        self.assertIn(BizCode.E_INTRINSIC_SINGULAR,
                      {f.code for f in result.failures})

    def test_illegal_fov_rejected(self) -> None:
        path = os.path.join(self.harness.sample_dir,
                            "view_04.camera.json")
        camera = read_json(path)
        camera["fov_deg"] = 200.0
        atomic_write_json(path, camera)
        result = load_sample(self.harness.sample_dir)
        self.assertIn(BizCode.E_FOV_ILLEGAL,
                      {f.code for f in result.failures})

    def test_resolution_mismatch_rejected(self) -> None:
        path = os.path.join(self.harness.sample_dir, "view_00.png")
        image = Image.new("RGB", (120, 90), (255, 255, 255))
        image.save(path, format="PNG")
        result = load_sample(self.harness.sample_dir)
        self.assertIn(BizCode.E_RESOLUTION_MISMATCH,
                      {f.code for f in result.failures})


if __name__ == "__main__":
    unittest.main()
