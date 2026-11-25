from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from layout_diag.config import load_config
from layout_diag.dataset import load_dataset
from layout_diag.generator import ImageSpec
from layout_diag.models import FailureLog
from layout_diag.validation import validate_workspace
from tests.helpers import make_dataset, read_config, write_config


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = make_dataset(
            cls.root / "samples",
            specs=[ImageSpec("small_layout_01.png", 2, 3, 0.0012, 5.0)],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def case(self, name: str) -> Path:
        destination = self.root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.samples, destination)
        return destination

    def test_valid_workspace(self) -> None:
        dataset, config, failures = validate_workspace(self.samples, self.samples / "config.json")
        self.assertFalse(failures.items)
        self.assertEqual(dataset.stats()["total_points"], 6)
        self.assertEqual(config.tier, "auto")

    def test_unknown_column_and_oob_keypoints(self) -> None:
        root = self.case("bad-keypoints")
        truth_path = root / "truth" / "small_layout_01.json"
        truth = json.loads(truth_path.read_text())
        truth["keypoints"][0]["column"] = "not-a-column"
        truth["keypoints"][1]["x"] = -1
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        failures = FailureLog()
        load_dataset(root, failures)
        codes = [failure.code for failure in failures.items]
        self.assertGreaterEqual(codes.count("E_KEYPOINT_OOB"), 2)
        self.assertTrue(any(item.details.get("reason") == "unknown_column" for item in failures.items))

    def test_overlapping_columns(self) -> None:
        root = self.case("overlap")
        truth_path = root / "truth" / "small_layout_01.json"
        truth = json.loads(truth_path.read_text())
        truth["columns"][1]["x0"] = 480
        truth["columns"][1]["x1"] = 530
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        failures = FailureLog()
        load_dataset(root, failures)
        self.assertIn("E_COLUMN_OVERLAP", [failure.code for failure in failures.items])

    def test_nonintegral_window_and_bad_quantization(self) -> None:
        config_path = self.root / "bad-config.json"
        config = read_config(self.samples / "config.json")
        config["window"]["width"] = 50
        config["precision"]["fixed_bits"] = 3
        config["precision"]["quant_step"] = 3
        write_config(config_path, config)
        failures = FailureLog()
        load_config(config_path, failures)
        codes = [failure.code for failure in failures.items]
        self.assertIn("E_WINDOW_ALIGN", codes)
        self.assertGreaterEqual(codes.count("E_QUANT_ENUM"), 2)

    def test_nonpositive_cc_configuration(self) -> None:
        config_path = self.root / "bad-cc.json"
        config = read_config(self.samples / "config.json")
        config["cc"]["min_area"] = 0
        config["cc"]["max_aspect"] = 0
        write_config(config_path, config)
        failures = FailureLog()
        load_config(config_path, failures)
        subjects = {failure.subject for failure in failures.items if failure.code == "E_CONFIG_VALUE"}
        self.assertIn("cc.min_area", subjects)
        self.assertIn("cc.max_aspect", subjects)

    def test_invalid_png_header_and_missing_active_version(self) -> None:
        root = self.case("bad-png")
        image_path = root / "images" / "small_layout_01.png"
        image_path.write_bytes(b"not-a-png")
        config_path = root / "config.json"
        config = read_config(config_path)
        config["keypoint_version"] = "v9"
        write_config(config_path, config)
        _, _, failures = validate_workspace(root, config_path)
        codes = [failure.code for failure in failures.items]
        self.assertIn("E_IMAGE_HEADER", codes)
        self.assertIn("E_KEYPOINT_VERSION", codes)

    def test_reading_sequence_must_start_at_zero(self) -> None:
        root = self.case("bad-reading")
        truth_path = root / "truth" / "small_layout_01.json"
        truth = json.loads(truth_path.read_text())
        for point in truth["keypoints"]:
            point["reading_index"] += 1
        for column in truth["columns"]:
            column["start_reading_index"] += 1
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        failures = FailureLog()
        load_dataset(root, failures)
        self.assertIn("E_READING_SEQUENCE", [failure.code for failure in failures.items])


if __name__ == "__main__":
    unittest.main()
