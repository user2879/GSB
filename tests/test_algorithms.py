from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from gsb_layout.algorithms import (
    confidence_details,
    detect_columns,
    match_block,
    regress_segment,
    segment_column,
    split_oversize,
)
from gsb_layout.config import load_config, with_overrides
from gsb_layout.io_layer import load_png, load_truth
from gsb_layout.sample_generator import generate_samples
from gsb_layout.templates import build_template_set


class TestLayoutAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.samples = Path(cls.temp.name) / "samples"
        generate_samples(cls.samples)
        cls.config = load_config(cls.samples / "config.json")
        cls.templates = build_template_set(cls.config.template_weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_column_detection_is_deterministic_and_right_to_left(self) -> None:
        image = load_png(self.samples / "page_001.png", self.config)
        truth = load_truth(self.samples / "page_001.png", image)
        first = detect_columns(image.pixels, self.config)
        second = detect_columns(image.pixels, self.config)
        self.assertEqual([c.as_dict() for c in first], [c.as_dict() for c in second])
        self.assertEqual(len(first), len(truth.columns))
        self.assertTrue(all(a.x0 > b.x0 for a, b in zip(first, first[1:])))
        for detected, expected in zip(first, truth.columns):
            tx0, _, tx1, _ = expected.bbox
            intersection = max(0, min(detected.x1, tx1) - max(detected.x0, tx0))
            union = max(detected.x1, tx1) - min(detected.x0, tx0)
            self.assertGreaterEqual(intersection / union, 0.65)

    def test_projection_segments_and_oversize_split_records(self) -> None:
        image = load_png(self.samples / "page_001.png", self.config)
        columns = detect_columns(image.pixels, self.config)
        found_split = False
        for column in columns:
            segments = segment_column(image.pixels, column, self.config)
            self.assertTrue(segments)
            self.assertTrue(all(segment.height > 0 for segment in segments))
            for segment in segments:
                regressed = regress_segment(image.pixels, column, segment, self.config)
                record = regressed["split"]
                if segment.height_class == "over_max":
                    found_split = True
                    self.assertTrue(record["applied"])
                    self.assertGreater(record["split_count"], 0)
                    self.assertEqual(len(record["segment_costs"]), record["parts"])
                else:
                    self.assertFalse(record["applied"])
                    self.assertEqual(record["penalty"], 0)
                for char in regressed["boxes"]:
                    self.assertEqual(len(char["bbox"]), 4)
                    self.assertIsInstance(char["residual"], int)
        self.assertTrue(found_split)

    def test_split_penalty_changes_real_cut_and_ties_choose_upper(self) -> None:
        projection = np.full(60, 1000, dtype=np.int64)
        projection[16] = 0
        projection[20] = 500
        low = with_overrides(
            self.config, split_penalty=0, char_min_height=10, char_max_height=30
        )
        high = with_overrides(
            self.config, split_penalty=100, char_min_height=10, char_max_height=30
        )
        low_solution = split_oversize(0, 40, projection, low)
        high_solution = split_oversize(0, 40, projection, high)
        self.assertIsNotNone(low_solution)
        self.assertIsNotNone(high_solution)
        self.assertNotEqual(low_solution.cuts, high_solution.cuts)

        symmetric = np.full(60, 1000, dtype=np.int64)
        symmetric[19] = 0
        symmetric[21] = 0
        tied = split_oversize(0, 40, symmetric, self.config)
        self.assertEqual(tied.cuts, (19,))
        self.assertTrue(tied.tied)

    def test_matching_tie_uses_code_point_order_and_confidence_flags(self) -> None:
        blank = np.zeros((20, 20), dtype=np.uint8)
        match = match_block(blank, [0, 0, 20, 20], self.config, self.templates)
        self.assertTrue(match["tie"])
        self.assertEqual(match["text"], min(match["tied_characters"], key=ord))
        self.assertEqual(match["tied_characters"], sorted(match["tied_characters"], key=ord))
        regression = {
            "residual": self.config.border_residual_tolerance + 5,
            "split_record": {"split_count": 1, "total_cost": 1000},
        }
        details = confidence_details(match, regression, self.config)
        self.assertTrue(details["low_confidence"])
        codes = {reason["code"] for reason in details["low_reasons"]}
        self.assertTrue({"TIE_BREAK", "BORDER_RESIDUAL", "SPLIT_BLOCK"} <= codes)
        self.assertGreaterEqual(details["confidence_q"], 0)
        self.assertLessEqual(details["confidence_q"], self.config.fixed_scale)
        self.assertEqual(
            details["operation_order"],
            [
                "score_q",
                "subtract_border_penalty",
                "subtract_split_penalty",
                "clip_confidence_interval",
            ],
        )


if __name__ == "__main__":
    unittest.main()
