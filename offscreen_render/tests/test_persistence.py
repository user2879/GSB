import os
import tempfile
import unittest

import numpy as np

from offscreen_render.errors import RenderError
from offscreen_render.persistence import (
    ARCHIVE_DIRNAME,
    compute_version_id,
    load_tile,
    save_tile,
)
from offscreen_render.renderer import TileResult
from offscreen_render.samplegen import build_deterministic_scene


def make_tile(index: int) -> TileResult:
    h = w = 4
    return TileResult(
        tile_index=index, x0=0, y0=0, width=w, height=h,
        grayscale=np.linspace(0, 1, h * w).reshape(h, w),
        diff_map=np.full((h, w), 1e-7),
        steps_map=np.full((h, w), 5, np.int32),
        reason_map=np.full((h, w), "completed", "<U16"),
        first_exceeds=[{"step": 1, "pixel": [0, 0]}],
        cutoff_count=2, max_steps_count=0,
    )


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        self.scene = build_deterministic_scene(self.state, tempfile.mkdtemp())
        self.version = compute_version_id(self.scene)

    def test_roundtrip_and_digest(self):
        tile = make_tile(0)
        path = save_tile(self.state, tile, self.version)
        self.assertTrue(os.path.exists(path))
        loaded = load_tile(self.state, 0, self.version)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded.grayscale, tile.grayscale)
        self.assertEqual(loaded.content_digest, tile.content_digest)

    def test_version_mismatch_archives(self):
        save_tile(self.state, make_tile(1), self.version)
        save_tile(self.state, make_tile(1), "v1-differentversion")
        archive = os.path.join(self.state, ARCHIVE_DIRNAME)
        self.assertTrue(os.path.isdir(archive))
        self.assertTrue(any("00001" in name for name in os.listdir(archive)))
        # 新版本仍可正常加载
        self.assertIsNotNone(load_tile(self.state, 1, "v1-differentversion"))
        self.assertIsNone(load_tile(self.state, 1, self.version))

    def test_corrupt_tile_raises_and_archives(self):
        save_tile(self.state, make_tile(2), self.version)
        path = [f for f in os.listdir(self.state) if f.startswith("tile_00002")][0]
        with open(os.path.join(self.state, path), "w", encoding="utf-8") as fh:
            fh.write("{corrupt")
        with self.assertRaises(RenderError) as ctx:
            load_tile(self.state, 2, self.version)
        self.assertEqual(ctx.exception.code, "E_3001")

    def test_atomic_no_tmp_left(self):
        save_tile(self.state, make_tile(3), self.version)
        leftovers = [f for f in os.listdir(self.state) if ".tmp." in f]
        self.assertEqual(leftovers, [])

    def test_deterministic_version_id(self):
        again = compute_version_id(
            build_deterministic_scene(self.state, tempfile.mkdtemp()))
        self.assertEqual(again, self.version)


if __name__ == "__main__":
    unittest.main()
