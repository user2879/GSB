from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import clone_bundle, make_sample

from inpaint_pyramid.dependencies import lower_tile_refs
from inpaint_pyramid.engine import IncrementalEngine, compare_engines
from inpaint_pyramid.pyramid import UPSAMPLE_METADATA, bilinear_upsample, block_mean_downsample


class PyramidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.bundle, cls.stream, _ = make_sample(Path(cls.temp.name) / "samples")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_block_mean_and_bilinear_are_deterministic(self) -> None:
        values = np.arange(4 * 4 * 3, dtype=np.int64).reshape(4, 4, 3)
        first = block_mean_downsample(values, 2)
        second = block_mean_downsample(values, 2)
        self.assertTrue(np.array_equal(first, second))
        up_first = bilinear_upsample(first, 4, 4)
        up_second = bilinear_upsample(first, 4, 4)
        self.assertTrue(np.array_equal(up_first, up_second))
        self.assertEqual(UPSAMPLE_METADATA["tie_break"], "half_up")

    def test_layer_fingerprints_and_full_rebuild_are_repeatable(self) -> None:
        first = IncrementalEngine.build_full(clone_bundle(self.bundle, threads=1))
        second = IncrementalEngine.build_full(clone_bundle(self.bundle, threads=4))
        self.assertEqual(first.layer_fingerprints(), second.layer_fingerprints())
        self.assertTrue(compare_engines(first, second)["equal"])
        self.assertTrue(all(fp is not None for fp in first.layer_fingerprints()))

    def test_lower_tile_refs_are_bounded_by_bilinear_support(self) -> None:
        for level in range(1, self.bundle.spec.levels):
            for row in range(self.bundle.spec.tile_rows(level)):
                for col in range(self.bundle.spec.tile_cols(level)):
                    refs = lower_tile_refs(self.bundle.spec, level, row, col)
                    self.assertTrue(refs)
                    self.assertTrue(all(ref[0] == level - 1 for ref in refs))


if __name__ == "__main__":
    unittest.main()

