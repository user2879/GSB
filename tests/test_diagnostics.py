from __future__ import annotations

import unittest

from glyphforge.diagnostics import region_metrics
from glyphforge.errors import ErrorCode, GlyphError


def square(x0: float, x1: float):
    return [[(x0, 0.0), (x1, 0.0), (x1, 10.0), (x0, 10.0)]]


class DiagnosticsTests(unittest.TestCase):
    def test_bitmap_difference_alone_does_not_raise(self) -> None:
        report = region_metrics(
            square(0.0, 10.0),
            square(0.05, 10.05),
            tolerance_a=0.1,
            tolerance_b=0.1,
            bitmap_size=32,
        )
        self.assertGreater(report["bitmap"]["differing_pixels"], 0)
        self.assertFalse(report["drift"])
        self.assertEqual(report["classification"], "WITHIN_TOLERANCE_BITMAP_DIFFERENCE")

    def test_geometry_drift_raises_business_code(self) -> None:
        with self.assertRaises(GlyphError) as context:
            region_metrics(
                square(0.0, 10.0),
                square(0.0, 20.0),
                tolerance_a=1e-5,
                tolerance_b=1e-5,
                bitmap_size=32,
            )
        self.assertEqual(context.exception.code, ErrorCode.CONSISTENCY_DRIFT)
        details = context.exception.details
        self.assertTrue(details["divergent_edges"])
        self.assertGreater(details["symmetric_difference_area"], details["error_interval"]["area_upper"])

    def test_identical_geometry_has_zero_metrics(self) -> None:
        loop = square(0.0, 10.0)
        report = region_metrics(loop, loop, tolerance_a=0.1, tolerance_b=0.1, bitmap_size=32)
        self.assertEqual(report["symmetric_difference_area"], 0.0)
        self.assertEqual(report["hausdorff_distance"], 0.0)
        self.assertEqual(report["bitmap"]["differing_pixels"], 0)
        self.assertFalse(report["drift"])


if __name__ == "__main__":
    unittest.main()
