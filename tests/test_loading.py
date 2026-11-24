from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quantcalib.errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_GROUP_INVALID,
    E_QUANT_BOUNDS,
    E_RECEIPT_ORPHAN,
    E_WEIGHT_DUPLICATE,
    E_WEIGHT_INVALID,
)
from quantcalib.loading import load_dataset
from quantcalib.samplegen import generate_samples

from tests.helpers import base_config, calibration_fixture, write_json, write_jsonl, write_yaml


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weights = self.root / "weights.jsonl"
        self.calibration = self.root / "calibration.json"
        self.config = self.root / "config.yaml"
        self.receipt = self.root / "receipts.jsonl"
        write_json(self.calibration, calibration_fixture())
        write_yaml(self.config, base_config())
        write_jsonl(
            self.weights,
            [
                {
                    "layer_name": "a",
                    "shape": [2, 2],
                    "weights": [1.0, -2.0, 3.0, -4.0],
                    "content_version": 1,
                    "group_granularity": 4,
                }
            ],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _load(self):
        return load_dataset(
            weights_path=self.weights,
            calibration_path=self.calibration,
            config_path=self.config,
            receipt_paths=(self.receipt,) if self.receipt.exists() else (),
        )

    def test_valid_dataset_and_aliases(self) -> None:
        dataset = self._load()
        self.assertFalse(dataset.failures.items)
        self.assertEqual(dataset.layers[0].numel, 4)
        self.assertEqual(dataset.calibration.sample_ids, ("s0", "s1"))

    def test_duplicate_layer(self) -> None:
        record = {
            "layer_name": "a",
            "shape": [2, 2],
            "weights": [1, 2, 3, 4],
            "content_version": 1,
            "group_granularity": 4,
        }
        write_jsonl(self.weights, [record, record])
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_WEIGHT_DUPLICATE))

    def test_nonfinite_and_version_order(self) -> None:
        write_jsonl(
            self.weights,
            [
                {
                    "layer_name": "a",
                    "shape": [2, 2],
                    "weights": [1, float("nan"), 3, 4],
                    "content_version": 2,
                    "group_granularity": 4,
                },
                {
                    "layer_name": "b",
                    "shape": [2, 2],
                    "weights": [1, 2, 3, 4],
                    "content_version": 1,
                    "group_granularity": 4,
                },
            ],
        )
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_WEIGHT_INVALID))

    def test_group_indivisible(self) -> None:
        write_jsonl(
            self.weights,
            [
                {
                    "layer_name": "a",
                    "shape": [2, 2],
                    "weights": [1, 2, 3, 4],
                    "content_version": 1,
                    "group_granularity": 3,
                }
            ],
        )
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_GROUP_INVALID))

    def test_config_bounds_and_percentiles(self) -> None:
        config = base_config()
        config["quantization"]["bit_width"] = 32
        config["quantization"]["percentile_lower_permille"] = 900
        config["quantization"]["percentile_upper_permille"] = 100
        write_yaml(self.config, config)
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_QUANT_BOUNDS))
        self.assertTrue(dataset.failures.has_code(E_CONFIG_INVALID))

    def test_calibration_dimension(self) -> None:
        calibration = calibration_fixture()
        calibration["samples"][0]["input"] = [1.0]
        write_json(self.calibration, calibration)
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_CALIBRATION_INVALID))

    def test_orphan_receipt(self) -> None:
        write_jsonl(
            self.receipt,
            [{"layer_name": "missing", "compression_permille": 500}],
        )
        dataset = self._load()
        self.assertTrue(dataset.failures.has_code(E_RECEIPT_ORPHAN))

    def test_three_receipt_versions_and_missing_markers(self) -> None:
        generated = generate_samples(self.root / "samples")
        dataset = load_dataset(
            weights_path=generated["weights"],
            calibration_path=generated["calibration"],
            config_path=generated["config"],
            receipt_paths=(generated["q1"], generated["q2"], generated["q3"]),
        )
        self.assertFalse(dataset.failures.items)
        by_version = {receipt.source_version: receipt for receipt in dataset.receipts.receipts}
        self.assertIn("parameter_version", by_version["q1"].missing_fields)
        self.assertIn("group_scale_digest", by_version["q2"].missing_fields)
        self.assertEqual(by_version["q3"].missing_fields, ())


if __name__ == "__main__":
    unittest.main()
