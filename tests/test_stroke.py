from __future__ import annotations

import math
import unittest

from glyphforge.errors import ErrorCode, GlyphError
from glyphforge.flatten import FlattenOptions, flatten_contours
from glyphforge.raster import check_bitmap_consistency, rasterize_stroke
from glyphforge.samples import generate_suite
from glyphforge.stroke import CapStyle, JoinStyle, StrokeOptions, stroke_polyline


class StrokeTests(unittest.TestCase):
    def test_all_caps_and_joins_are_deterministic_and_bitmap_consistent(self) -> None:
        suite = generate_suite()
        case = suite.by_id("G04_OPEN_PATH")
        flat = flatten_contours((case.open_path,), FlattenOptions(24), source="open")
        for cap in CapStyle:
            for join in JoinStyle:
                options = StrokeOptions(case.stroke_width, cap, join, flat.tolerance)
                first = stroke_polyline(flat.open_points, options)
                second = stroke_polyline(flat.open_points, options)
                self.assertEqual(first.boundary_hash(), second.boundary_hash())
                boundary, direct = rasterize_stroke(first, 24)
                report = check_bitmap_consistency(
                    boundary,
                    direct,
                    boundary_loops=first.loops,
                    direct_loops=first.direct_loops,
                    declared_tolerance=flat.tolerance,
                )
                self.assertTrue(report.consistent)
                self.assertLessEqual(first.arc_approximation_error, flat.tolerance + 1e-12)

    def test_cap_area(self) -> None:
        points = [(0.0, 0.0), (10.0, 0.0)]
        butt = stroke_polyline(points, StrokeOptions(2.0, "butt", "bevel", 0.01))
        square = stroke_polyline(points, StrokeOptions(2.0, "square", "bevel", 0.01))
        self.assertAlmostEqual(butt.boolean_result.area, 20.0, places=8)
        self.assertAlmostEqual(square.boolean_result.area, 24.0, delta=0.01)

    def test_business_errors(self) -> None:
        with self.assertRaises(GlyphError) as width_error:
            stroke_polyline([(0, 0), (1, 0)], StrokeOptions(0.0))
        self.assertEqual(width_error.exception.code, ErrorCode.STROKE_WIDTH)
        with self.assertRaises(GlyphError) as cap_error:
            stroke_polyline([(0, 0), (1, 0)], StrokeOptions(1.0, cap="triangle"))
        self.assertEqual(cap_error.exception.code, ErrorCode.CAP_JOIN)
        with self.assertRaises(GlyphError) as join_error:
            stroke_polyline([(0, 0), (1, 0)], StrokeOptions(1.0, join="round_bad"))
        self.assertEqual(join_error.exception.code, ErrorCode.CAP_JOIN)


if __name__ == "__main__":
    unittest.main()
