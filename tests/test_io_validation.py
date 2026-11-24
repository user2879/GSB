from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from gsb_layout.config import load_config
from gsb_layout.pipeline import Pipeline
from gsb_layout.sample_generator import generate_samples


class TestInputValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = self.root / "samples"
        generate_samples(self.samples)
        self.config_path = self.samples / "config.json"
        self.config = load_config(self.config_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, state_name: str = "state") -> dict:
        return Pipeline(self.samples, self.config, self.root / state_name).run(full_rebuild=True)

    def test_bad_images_and_truth_do_not_interrupt_other_images(self) -> None:
        # Add a fourth valid page before breaking the first three files.
        shutil.copy2(self.samples / "page_003.png", self.samples / "page_004.png")
        truth = json.loads((self.samples / "page_003.truth.json").read_text(encoding="utf-8"))
        truth["image"] = "page_004.png"
        (self.samples / "page_004.truth.json").write_text(
            json.dumps(truth, ensure_ascii=False), encoding="utf-8"
        )
        (self.samples / "page_001.png").write_bytes(b"not a png")
        (self.samples / "page_002.truth.json").unlink()
        broken = json.loads((self.samples / "page_003.truth.json").read_text(encoding="utf-8"))
        broken["columns"][0]["chars"][0]["bbox"][0] = 0.5
        (self.samples / "page_003.truth.json").write_text(
            json.dumps(broken, ensure_ascii=False), encoding="utf-8"
        )
        report = self._run()
        self.assertEqual([image["name"] for image in report["images"]], ["page_004.png"])
        self.assertEqual(
            [failure["code"] for failure in report["failures"]],
            ["E_IMAGE_INVALID", "E_TRUTH_INVALID", "E_TRUTH_INVALID"],
        )
        self.assertEqual(
            [failure["failure_id"] for failure in report["failures"]],
            ["F0001", "F0002", "F0003"],
        )
        self.assertEqual(report["summary"]["failed_images"], 3)

    def test_truth_overlap_and_empty_text_are_rejected(self) -> None:
        raw = json.loads((self.samples / "page_001.truth.json").read_text(encoding="utf-8"))
        first = raw["columns"][0]["chars"][0]["bbox"]
        second = raw["columns"][0]["chars"][1]["bbox"]
        second[1] = first[1]
        second[3] = first[3]
        (self.samples / "page_001.truth.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        report = self._run("state_overlap")
        self.assertIn(
            "E_TRUTH_INVALID",
            [failure["code"] for failure in report["failures"]],
        )
        self.assertEqual(len(report["images"]), 2)

        raw = json.loads((self.samples / "page_002.truth.json").read_text(encoding="utf-8"))
        raw["columns"][0]["chars"][0]["text"] = " "
        (self.samples / "page_002.truth.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        report = self._run("state_text")
        self.assertEqual(
            [failure["code"] for failure in report["failures"]],
            ["E_TRUTH_INVALID", "E_TRUTH_INVALID"],
        )

    def test_out_of_range_bbox_is_rejected(self) -> None:
        raw = json.loads((self.samples / "page_001.truth.json").read_text(encoding="utf-8"))
        raw["columns"][0]["chars"][0]["bbox"][2] = self.config.expected_width + 10
        (self.samples / "page_001.truth.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        report = self._run()
        self.assertEqual(report["failures"][0]["code"], "E_TRUTH_INVALID")
        self.assertEqual(report["failures"][0]["image"], "page_001.png")


if __name__ == "__main__":
    unittest.main()
