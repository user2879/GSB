import hashlib
import tempfile
import unittest
from pathlib import Path

from subtitle_workbench.errors import WrapError
from subtitle_workbench.models import canonical_bytes
from subtitle_workbench.sample import DEFAULT_SEED, generate_sample, write_sample
from subtitle_workbench.wrapping import (FONT_TIERS, glyph_width, wrap_text)


class SampleGenerationTests(unittest.TestCase):
    def test_four_required_situations_present(self):
        project = generate_sample()
        overlaps = [len(row["overlaps_with"]) for row in project.manifest]
        self.assertGreaterEqual(max(overlaps), 3)  # 4-way cluster
        self.assertTrue(any(row["crosses_cut"] for row in project.manifest))
        self.assertTrue(any(item.is_ad for item in project.items))
        # long texts force at least one downgrade below the largest tier
        long_items = [item for item in project.items
                      if len(item.text) >= 40]
        self.assertGreaterEqual(len(long_items), 3)

    def test_all_intervals_valid_and_inside_timeline(self):
        project = generate_sample()
        for item in project.items:
            self.assertLess(item.start_frame, item.end_frame)
            self.assertGreaterEqual(item.start_frame, 0)
            self.assertLessEqual(item.end_frame,
                                 project.timeline.total_frames)

    def test_byte_identical_across_runs(self):
        first = canonical_bytes(generate_sample(DEFAULT_SEED).to_dict())
        second = canonical_bytes(generate_sample(DEFAULT_SEED).to_dict())
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.json"
            p2 = Path(tmp) / "b.json"
            write_sample(generate_sample(), p1)
            write_sample(generate_sample(), p2)
            self.assertEqual(hashlib.sha256(p1.read_bytes()).digest(),
                             hashlib.sha256(p2.read_bytes()).digest())

    def test_ground_truth_matches_items(self):
        project = generate_sample()
        by_id = {item.item_id: item for item in project.items}
        for row in project.manifest:
            item = by_id[row["item_id"]]
            self.assertEqual(item.start_frame, row["start_frame"])
            self.assertEqual(item.end_frame, row["end_frame"])
            self.assertEqual(item.is_ad, row["is_ad"])


class WrappingTests(unittest.TestCase):
    def test_word_boundary_preferred_over_glyph_break(self):
        result = wrap_text("alpha beta gamma delta epsilon zeta",
                           safe_width=400, side_margin=20, max_lines=3)
        for line in result.lines:
            self.assertTrue(line.startswith(("alpha", "beta", "gamma",
                                             "delta", "epsilon", "zeta")))
        self.assertFalse(result.hard_broken)

    def test_no_line_exceeds_safe_width(self):
        text = ("deterministic wrapping " * 8) + "中文混排" * 12
        result = wrap_text(text)
        self.assertLessEqual(result.max_line_width, 1920 - 2 * 120 + 1e-9)
        for line in result.lines:
            self.assertLessEqual(
                sum(glyph_width(ch) for ch in line)
                * result.font_size / 100.0,
                1920 - 2 * 120 + 1e-9)

    def test_font_downgrade_is_deterministic(self):
        text = "字" * 120
        r1 = wrap_text(text)
        r2 = wrap_text(text)
        self.assertEqual(r1, r2)
        self.assertLess(r1.font_size, FONT_TIERS[0])

    def test_impossible_width_raises_business_code(self):
        with self.assertRaises(WrapError) as ctx:
            wrap_text("一" * 10, safe_width=40, side_margin=18,
                      font_tiers=(10,))
        self.assertEqual(ctx.exception.code, "E_WRAP_IMPOSSIBLE")

    def test_cjk_hard_break_keeps_width_contract(self):
        # A single very long CJK run cannot satisfy one line, so the smallest
        # tier hard-breaks it while every line still respects the safe width.
        result = wrap_text("中" * 200, max_lines=1)
        self.assertGreater(len(result.lines), 1)
        self.assertLessEqual(result.max_line_width, 1680 + 1e-9)


if __name__ == "__main__":
    unittest.main()
