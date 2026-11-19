from __future__ import annotations

import unittest

from glyphforge.errors import ErrorCode, GlyphError
from glyphforge.model import Contour, OutlineNode, Point, SegmentType
from glyphforge.samples import generate_suite


class SampleTests(unittest.TestCase):
    def test_two_generations_are_byte_identical(self) -> None:
        first = generate_suite(12345)
        second = generate_suite(12345)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.content_hash(), second.content_hash())
        self.assertNotEqual(first.content_hash(), generate_suite(54321).content_hash())

    def test_required_categories_and_sizes(self) -> None:
        suite = generate_suite()
        self.assertEqual(
            {case.category for case in suite.cases},
            {"simple", "self_intersect", "overlap_coincident", "open_path", "degenerate"},
        )
        self.assertEqual(len({case.glyph_id for case in suite.cases}), 5)
        for case in suite.cases:
            self.assertEqual(case.suggested_sizes, (12, 16, 24, 32, 48))

    def test_validation_rejects_wrong_control_count(self) -> None:
        contour = Contour(
            (
                OutlineNode(Point(0, 0), SegmentType.MOVE),
                OutlineNode(Point(1, 0), SegmentType.LINE, (Point(0.5, 0.5),)),
            )
        )
        with self.assertRaises(GlyphError) as context:
            contour.validate()
        self.assertEqual(context.exception.code, ErrorCode.OUTLINE_INVALID)

    def test_validation_rejects_non_finite_and_open_closure(self) -> None:
        with self.assertRaises(GlyphError) as context:
            Contour(
                (
                    OutlineNode(Point(0, 0), SegmentType.MOVE),
                    OutlineNode(Point(float("nan"), 0), SegmentType.LINE, (), True),
                )
            ).validate(require_closed=True)
        self.assertEqual(context.exception.code, ErrorCode.OUTLINE_INVALID)

        with self.assertRaises(GlyphError):
            Contour(
                (
                    OutlineNode(Point(0, 0), SegmentType.MOVE),
                    OutlineNode(Point(1, 0), SegmentType.LINE),
                    OutlineNode(Point(0, 1), SegmentType.LINE, (), True),
                )
            ).validate(require_closed=True)


if __name__ == "__main__":
    unittest.main()
