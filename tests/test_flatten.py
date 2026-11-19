from __future__ import annotations

import math
import unittest

from glyphforge.flatten import FlattenOptions, flatten_contours
from glyphforge.samples import generate_suite


class FlattenTests(unittest.TestCase):
    def test_deterministic_and_monotone_with_tolerance(self) -> None:
        suite = generate_suite()
        for case in suite.cases:
            previous = -1
            for scale in (4.0, 2.0, 1.0, 0.5, 0.25):
                first = flatten_contours(
                    case.source_a,
                    FlattenOptions(24, tolerance_scale=scale),
                    source="a",
                )
                second = flatten_contours(
                    case.source_a,
                    FlattenOptions(24, tolerance_scale=scale),
                    source="a",
                )
                self.assertEqual(first.content_hash(), second.content_hash())
                self.assertGreaterEqual(first.vertex_count, previous)
                previous = first.vertex_count
                for contour in first.contours:
                    for segment in contour.segments:
                        self.assertTrue(math.isfinite(segment.estimated_error))

    def test_derived_tolerance_tracks_pixel_size(self) -> None:
        coarse = FlattenOptions(12).tolerance()
        fine = FlattenOptions(48).tolerance()
        self.assertGreater(coarse, fine)
        self.assertAlmostEqual(fine, coarse / 4.0)

    def test_degenerate_segments_are_marked_and_skipped(self) -> None:
        suite = generate_suite()
        case = suite.by_id("G05_DEGENERATE")
        flat = flatten_contours(case.source_a, FlattenOptions(24), source="a")
        degenerate = [
            segment
            for contour in flat.contours
            for segment in contour.segments
            if segment.degenerate
        ]
        self.assertTrue(degenerate)
        self.assertTrue(all(segment.vertex_count == 0 for segment in degenerate))
        self.assertTrue(all(segment.reason in {"zero_length", "coincident_control_points"} for segment in degenerate))
        for contour in flat.contours:
            self.assertGreaterEqual(len(contour.points), 3)
            for x, y in contour.points:
                self.assertTrue(math.isfinite(x) and math.isfinite(y))


if __name__ == "__main__":
    unittest.main()
