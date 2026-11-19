from __future__ import annotations

import unittest

from glyphforge.boolean import BooleanOp, ResolutionOrder, boolean_operation
from glyphforge.flatten import FlattenOptions, flatten_contours
from glyphforge.geometry import signed_area
from glyphforge.raster import check_bitmap_consistency, rasterize_boolean
from glyphforge.samples import generate_suite


def square(x0: float, y0: float, x1: float, y1: float):
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


class BooleanTests(unittest.TestCase):
    def test_overlapping_square_areas(self) -> None:
        a = square(0, 0, 10, 10)
        b = square(5, 0, 15, 10)
        expected = {
            BooleanOp.UNION: 150.0,
            BooleanOp.INTERSECTION: 50.0,
            BooleanOp.DIFFERENCE: 50.0,
            BooleanOp.XOR: 100.0,
        }
        for op, area in expected.items():
            result = boolean_operation(a, b, op)
            self.assertAlmostEqual(result.area, area, places=8)
            self.assertTrue(all(signed_area(loop) > 0.0 for loop in result.loops))

    def test_coincident_edges_have_unique_resolution(self) -> None:
        a = square(0, 0, 10, 10)
        b = square(0, 0, 10, 10)
        self.assertAlmostEqual(boolean_operation(a, b, BooleanOp.UNION).area, 100.0)
        self.assertAlmostEqual(boolean_operation(a, b, BooleanOp.INTERSECTION).area, 100.0)
        self.assertEqual(boolean_operation(a, b, BooleanOp.DIFFERENCE).loops, ())
        self.assertEqual(boolean_operation(a, b, BooleanOp.XOR).loops, ())

    def test_resolution_order_is_deterministic(self) -> None:
        suite = generate_suite()
        for case in suite.cases:
            flat_a = flatten_contours(case.source_a, FlattenOptions(24), source="a")
            flat_b = flatten_contours(case.source_b, FlattenOptions(24), source="b")
            for op in BooleanOp:
                first = boolean_operation(flat_a.loops, flat_b.loops, op)
                second = boolean_operation(flat_a.loops, flat_b.loops, op)
                other_order = boolean_operation(
                    flat_a.loops,
                    flat_b.loops,
                    op,
                    resolution_order=ResolutionOrder.SOURCE_THEN_LEX,
                )
                self.assertEqual(first.boundary_hash(), second.boundary_hash())
                self.assertEqual(first.boundary_hash(), other_order.boundary_hash())

    def test_boolean_bitmap_paths_match_for_all_samples(self) -> None:
        suite = generate_suite()
        for case in suite.cases:
            for size in (12, 24, 48):
                flat_a = flatten_contours(case.source_a, FlattenOptions(size), source="a")
                flat_b = flatten_contours(case.source_b, FlattenOptions(size), source="b")
                for op in BooleanOp:
                    result = boolean_operation(flat_a.loops, flat_b.loops, op)
                    boundary, direct = rasterize_boolean(result, size)
                    report = check_bitmap_consistency(
                        boundary,
                        direct,
                        boundary_loops=result.loops,
                        direct_loops=result.direct_loops,
                        declared_tolerance=flat_a.tolerance,
                    )
                    self.assertTrue(report.consistent, msg=f"{case.glyph_id} {size} {op.value}")


if __name__ == "__main__":
    unittest.main()
