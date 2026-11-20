"""测试公共夹具：在临时目录生成确定性小样本。"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from vrep.samplegen import generate_all


class SampleTestBase(unittest.TestCase):
    IMAGE_SIZE = 64
    COUNT = 8
    TILE_SIZE = 16

    def setUp(self) -> None:
        self.temp_root = tempfile.mkdtemp(prefix="vrep-test-")
        self.samples = os.path.join(self.temp_root, "samples")
        summary = generate_all(
            self.samples,
            image_size=self.IMAGE_SIZE,
            count=self.COUNT,
            seed=20261121,
        )
        self.tile_size = summary["tile_size"]
        self.workspace = os.path.join(self.temp_root, "workspace")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def config_path(self, name: str = "config.json") -> str:
        return os.path.join(self.samples, name)
