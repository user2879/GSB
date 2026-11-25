from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from layout_diag.components import _label_raw_components, run_component_stage
from layout_diag.adaptive import select_effective_config
from layout_diag.config import load_config
from layout_diag.generator import ImageSpec
from layout_diag.sliding import run_sliding_stage
from tests.helpers import make_dataset


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = make_dataset(
            cls.root / "samples",
            specs=[ImageSpec("medium_layout_01.png", 3, 6, 0.002, 4.0)],
        )
        cls.config = load_config(cls.samples / "config.json")
        with Image.open(cls.samples / "images" / "medium_layout_01.png") as image:
            cls.grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        cls.image_fp = "a" * 64
        cls.small_effective, _ = select_effective_config(
            cls.config,
            {"image_count": 1, "total_points": 6, "max_columns": 2, "max_chars_per_column": 3},
        )
        cls.medium_effective, _ = select_effective_config(
            cls.config,
            {"image_count": 1, "total_points": 18, "max_columns": 3, "max_chars_per_column": 6},
        )
        cls.large_effective, _ = select_effective_config(
            cls.config,
            {"image_count": 5, "total_points": 180, "max_columns": 5, "max_chars_per_column": 10},
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_sliding_is_repeatable_and_window_dependent(self) -> None:
        first = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, self.small_effective)
        second = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, self.small_effective)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        other = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, self.medium_effective)
        self.assertNotEqual(first["candidate_count"], other["candidate_count"])
        self.assertNotEqual(
            [(item["window_id"], item["score_q"]) for item in first["candidates"]],
            [(item["window_id"], item["score_q"]) for item in other["candidates"]],
        )
        self.assertTrue(all("window_id" in item and item["window_id"] for item in first["candidates"]))

    def test_components_record_merge_split_and_filters(self) -> None:
        sliding = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, self.medium_effective)
        result = run_component_stage(
            "medium_layout_01.png",
            self.grayscale,
            self.medium_effective,
            sliding["candidates"],
        )
        self.assertGreater(len(result["merge_records"]), 0)
        self.assertTrue(all(record["basis"] for record in result["merge_records"]))
        self.assertTrue(
            all(any("merge_gap" in reason for reason in record["basis"]) for record in result["merge_records"])
        )
        self.assertEqual(result["component_count"], 18)
        self.assertTrue(all(component["window_id"] for component in result["components"]))
        self.assertTrue(result["filtered"])

    def test_tier_selector_capacity_and_real_path_change(self) -> None:
        small, small_basis = select_effective_config(
            self.config,
            {"image_count": 1, "total_points": 6, "max_columns": 2, "max_chars_per_column": 3},
        )
        medium, _ = select_effective_config(
            self.config,
            {"image_count": 1, "total_points": 18, "max_columns": 3, "max_chars_per_column": 6},
        )
        large, _ = select_effective_config(
            self.config,
            {"image_count": 3, "total_points": 74, "max_columns": 5, "max_chars_per_column": 10},
        )
        self.assertEqual((small.tier, medium.tier, large.tier), ("small", "medium", "large"))
        self.assertEqual(small_basis["mode"], "auto")
        small_slide = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, small)
        large_slide = run_sliding_stage("medium_layout_01.png", self.image_fp, self.grayscale, large)
        self.assertNotEqual(small_slide["candidate_count"], large_slide["candidate_count"])
        self.assertNotEqual(small.precision.as_dict(), large.precision.as_dict())
        self.assertNotEqual(small.cc.merge_gap, large.cc.merge_gap)

    def test_merge_threshold_can_join_split_fragments(self) -> None:
        ink = np.zeros((40, 80), dtype=bool)
        ink[5:15, 10:20] = True
        ink[17:27, 10:20] = True
        raw = _label_raw_components(ink)
        self.assertEqual(len(raw), 2)
        with_gap = replace(self.medium_effective, cc=replace(self.medium_effective.cc, merge_gap=1))
        merged = replace(self.medium_effective, cc=replace(self.medium_effective.cc, merge_gap=3))
        low = run_component_stage("fragments.png", np.where(ink, 0, 255).astype(np.uint8), with_gap, [])
        high = run_component_stage("fragments.png", np.where(ink, 0, 255).astype(np.uint8), merged, [])
        self.assertEqual(low["component_count"], 0)  # no candidate overlap
        self.assertEqual(high["component_count"], 0)
        self.assertEqual(len(low["merge_records"]), 0)
        self.assertGreaterEqual(len(high["merge_records"]), 1)


if __name__ == "__main__":
    unittest.main()
