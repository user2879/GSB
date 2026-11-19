from __future__ import annotations

import math
import unittest

from diffrender.renderer import (
    compare_gradient_to_finite_difference,
    render_full,
    tile_layout,
)
from diffrender.scene import default_scene, target_scene_from


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = default_scene()
        cls.target_scene = target_scene_from(cls.scene)
        cls.first = render_full(cls.scene)
        cls.second = render_full(cls.scene)
        cls.target = render_full(cls.target_scene).image

    def test_full_render_is_byte_deterministic(self) -> None:
        self.assertEqual(self.first.image_hash, self.second.image_hash)
        self.assertEqual(self.first.raster_hash, self.second.raster_hash)
        self.assertEqual(self.first.shade_hash, self.second.shade_hash)
        self.assertEqual(self.first.render_hash, self.second.render_hash)
        for ty, tx, *_ in tile_layout(self.scene):
            a = self.first.tiles[(ty, tx)]
            b = self.second.tiles[(ty, tx)]
            self.assertEqual(a.ids, b.ids)
            self.assertEqual(a.bary, b.bary)
            self.assertEqual(a.depth, b.depth)
            self.assertEqual(a.coverage, b.coverage)
            self.assertEqual(a.normals, b.normals)
            self.assertEqual(a.intensity, b.intensity)
            self.assertEqual(a.content_hash(), b.content_hash())

    def test_raster_buffers_have_expected_layout(self) -> None:
        seen_spheres: set[int] = set()
        for ty, tx, *_ in tile_layout(self.scene):
            tile = self.first.tiles[(ty, tx)]
            self.assertEqual(len(tile.ids), self.scene.tile_size * self.scene.tile_size)
            self.assertEqual(len(tile.bary), 3 * len(tile.ids))
            self.assertEqual(len(tile.depth), len(tile.ids))
            self.assertEqual(len(tile.coverage), len(tile.ids))
            seen_spheres.update(index for index in tile.spheres)
            for pixel, sphere_id in enumerate(tile.ids):
                if sphere_id < 0:
                    continue
                bary = tile.bary[pixel * 3 : pixel * 3 + 3]
                self.assertAlmostEqual(sum(bary), 1.0, places=12)
                self.assertGreaterEqual(tile.coverage[pixel], 0.0)
                self.assertLessEqual(tile.coverage[pixel], 1.0)
                self.assertLess(tile.depth[pixel], 1.0e100)
                self.assertEqual(tile.spheres, tuple(sorted(tile.spheres)))
        self.assertEqual(seen_spheres, {0, 1, 2})
        self.assertTrue(all(math.isfinite(value) for value in self.first.image))

    def test_analytic_gradients_match_finite_differences(self) -> None:
        report = compare_gradient_to_finite_difference(
            self.scene, self.target, tolerance=2.0e-4, step=1.0e-8
        )
        self.assertEqual(set(report), set(self.scene.param_keys()))
        self.assertLess(max(item["abs_error"] for item in report.values()), 1.0e-6)


if __name__ == "__main__":
    unittest.main()

