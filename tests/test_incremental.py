from __future__ import annotations

import unittest

from glyphforge.incremental import IncrementalEngine
from glyphforge.model import Point
from glyphforge.samples import generate_suite


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = IncrementalEngine(generate_suite())
        cls.engine.build_all()

    def test_control_point_closure_and_full_rebuild(self) -> None:
        engine = self.engine
        before = engine.all_glyph_hashes()
        control = engine.cases["G02_SELF_INTERSECT"].source_a[1].nodes[1].controls[0]
        closure = engine.edit_control_point(
            "G02_SELF_INTERSECT",
            "a",
            1,
            1,
            0,
            Point(control.x + 0.2, control.y - 0.1),
        )
        actual = engine.recompute(closure)
        self.assertEqual(actual, closure)
        after = engine.all_glyph_hashes()
        self.assertNotEqual(before["G02_SELF_INTERSECT"], after["G02_SELF_INTERSECT"])
        for glyph_id in before:
            if glyph_id != "G02_SELF_INTERSECT":
                self.assertEqual(before[glyph_id], after[glyph_id])
        self.assertEqual(
            engine.glyph_summary("G02_SELF_INTERSECT")["content_hash"],
            engine.recompute_glyph_full("G02_SELF_INTERSECT")["content_hash"],
        )

    def test_stroke_change_does_not_recompute_booleans(self) -> None:
        engine = self.engine
        before = engine.all_glyph_hashes()
        closure = engine.set_stroke_parameters("G04_OPEN_PATH", cap="square", join="bevel")
        self.assertFalse(any(task.startswith("bool/") for task in closure))
        actual = engine.recompute(closure)
        self.assertEqual(actual, closure)
        after = engine.all_glyph_hashes()
        self.assertNotEqual(before["G04_OPEN_PATH"], after["G04_OPEN_PATH"])
        self.assertEqual(
            engine.glyph_summary("G04_OPEN_PATH")["content_hash"],
            engine.recompute_glyph_full("G04_OPEN_PATH")["content_hash"],
        )


if __name__ == "__main__":
    unittest.main()
