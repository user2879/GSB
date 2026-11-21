from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from candidate_exchange.config import load_detection_params, load_match_config
from candidate_exchange.conversion import run_conversion
from candidate_exchange.v1 import V1_ROW_WIDTH, parse_v1
from candidate_exchange.v2 import parse_v2

from tests.support import ensure_sample_data


class ParsingConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ensure_sample_data()

    def test_fixed_width_v1_and_v2_parse_all_rows(self) -> None:
        v1 = parse_v1(self.data / "candidates_v1.txt")
        v2 = parse_v2(self.data / "candidates_v2.json")
        v1_count = sum(len(items) for items in v1.frames.values())
        v2_count = sum(len(items) for items in v2.frames.values())
        self.assertEqual(v1_count, 99)
        self.assertEqual(v2_count, 99)
        self.assertFalse(v1.failures)
        self.assertFalse(v2.failures)
        for line in (self.data / "candidates_v1.txt").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                self.assertEqual(len(line), V1_ROW_WIDTH)

    def test_truncated_v1_line_reports_field_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(self.data / "candidates_v1.txt", root / "candidates_v1.txt")
            lines = (root / "candidates_v1.txt").read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1][:31]
            (root / "candidates_v1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            parsed = parse_v1(root / "candidates_v1.txt")
            self.assertTrue(parsed.failures)
            self.assertEqual(parsed.failures[0].code, "E_FIELD_TRUNCATED")

    def test_bidirectional_conversion_has_residual_sidecar_and_histograms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            conversion = run_conversion(
                v1=parse_v1(self.data / "candidates_v1.txt"),
                v2=parse_v2(self.data / "candidates_v2.json"),
                detection=load_detection_params(self.data),
                match=load_match_config(self.data),
                output_dir=output,
            )
            v1_to_v2 = conversion["checks"]["v1_to_v2"]
            v2_to_v1 = conversion["checks"]["v2_to_v1"]
            self.assertEqual(v1_to_v2["max_coordinate_deviation"], 0.0)
            self.assertEqual(v1_to_v2["max_score_deviation"], 0.0)
            self.assertGreater(v2_to_v1["max_coordinate_deviation"], 0.0)
            self.assertGreater(v2_to_v1["max_score_deviation"], 0.0)
            self.assertGreater(len(v2_to_v1["coordinate_abs_error_histogram"]), 1)
            self.assertGreater(len(v2_to_v1["score_abs_error_histogram"]), 1)
            self.assertTrue(v1_to_v2["class_mapping"]["complete"])
            self.assertTrue(v2_to_v1["class_mapping"]["complete"])
            self.assertTrue(Path(conversion["outputs"]["sidecar"]).exists())


if __name__ == "__main__":
    unittest.main()
