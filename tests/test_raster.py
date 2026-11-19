from __future__ import annotations

import unittest

from glyphforge.raster import Viewport, check_bitmap_consistency, rasterize_loops


class RasterTests(unittest.TestCase):
    def test_square_coverage(self) -> None:
        bitmap = rasterize_loops([[(0, 0), (10, 0), (10, 10), (0, 10)]], 32)
        # 100 glyph-square units / (64*64/32/32) = 25 full pixels.
        self.assertAlmostEqual(bitmap.coverage_total, 25.0, delta=0.1)
        self.assertGreater(bitmap.nonzero_pixels, 0)
        self.assertEqual(bitmap.data[0], 0)

    def test_hole_is_subtracted(self) -> None:
        outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (5, 15), (15, 15), (15, 5)]
        bitmap = rasterize_loops([outer, hole], 32)
        expected = (400 - 100) / 4.0
        self.assertAlmostEqual(bitmap.coverage_total, expected, delta=0.15)

    def test_mismatch_reports_first_pixel_and_winding(self) -> None:
        a = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
        b = [[(0, 0), (11, 0), (11, 10), (0, 10)]]
        bitmap_a = rasterize_loops(a, 32)
        bitmap_b = rasterize_loops(b, 32)
        report = check_bitmap_consistency(
            bitmap_a,
            bitmap_b,
            boundary_loops=a,
            direct_loops=b,
            declared_tolerance=0.1,
        )
        self.assertFalse(report.consistent)
        self.assertIsNotNone(report.first_divergence)
        assert report.first_divergence is not None
        self.assertIn("boundary_winding", report.first_divergence)
        self.assertIn("ray_intersections", report.first_divergence)
        self.assertEqual(report.first_divergence["tolerance"], 0.1)


if __name__ == "__main__":
    unittest.main()
