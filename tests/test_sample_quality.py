from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gsb_layout.config import load_config
from gsb_layout.pipeline import Pipeline
from gsb_layout.sample_generator import generate_samples


def _intersection_over_union(a: list[int], b: list[int]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


class TestSampleQuality(unittest.TestCase):
    def test_detection_regression_confidence_and_truth_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            samples = Path(temp) / "samples"
            generate_samples(samples)
            config = load_config(samples / "config.json")
            report = Pipeline(samples, config, Path(temp) / "state").run(full_rebuild=True)
            self.assertEqual(report["failures"], [])
            total = correct = 0
            split_chars = 0
            for image in report["images"]:
                truth = image["truth"]
                self.assertEqual(len(image["columns"]), len(truth["columns"]))
                for column, truth_column in zip(image["columns"], truth["columns"]):
                    self.assertGreater(len(column["chars"]), 0)
                    self.assertEqual(len(column["chars"]), len(truth_column["chars"]))
                    for char, truth_char in zip(column["chars"], truth_column["chars"]):
                        total += 1
                        if char["text"] == truth_char["text"]:
                            correct += 1
                        iou = _intersection_over_union(char["bbox"], truth_char["bbox"])
                        self.assertGreaterEqual(
                            iou,
                            0.55,
                            msg=f"{image['name']} col {column['reading_index']} {char['bbox']} vs {truth_char['bbox']}",
                        )
                        self.assertGreaterEqual(char["confidence_q"], 0)
                        self.assertLessEqual(char["confidence_q"], config.fixed_scale)
                        detail = char["confidence_detail"]
                        self.assertEqual(detail["confidence_q"], char["confidence_q"])
                        if char["split_record"]["split_count"] > 0:
                            split_chars += 1
                            self.assertTrue(char["low_confidence"])
                            codes = {reason["code"] for reason in char["low_reasons"]}
                            self.assertIn("SPLIT_BLOCK", codes)
            self.assertGreaterEqual(correct / total, 0.90)
            self.assertGreater(split_chars, 0)
            self.assertGreater(
                report["low_confidence_reason_counts"].get("SPLIT_BLOCK", 0), 0
            )


if __name__ == "__main__":
    unittest.main()
