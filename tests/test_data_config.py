from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from tabular_lab.config import load_config
from tabular_lab.data import generate_dataset
from tabular_lab.errors import (
    E_CONFIG_INVALID,
    E_MIX_RANGE,
    E_SCHEME_UNKNOWN,
    LabError,
)
from tests.helpers import BASE_CONFIG, LabFixture


class DataConfigTests(unittest.TestCase):
    def test_deterministic_generation_and_missing_fills(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            config = load_config(fixture.config_path)
            first = generate_dataset(config.data, config.seed)
            second = generate_dataset(config.data, config.seed)
            self.assertEqual(first.data_fingerprint, second.data_fingerprint)
            self.assertEqual(first.matrix_standardized.tobytes(), second.matrix_standardized.tobytes())

            self.assertTrue(np.isnan(first.raw_columns["income"]).any())
            income_index = first.feature_indices["income"]
            mean = first.numeric_fill_means["income"]
            expected_income = first.raw_columns["income"].astype(float).copy()
            expected_income[np.isnan(expected_income)] = mean
            np.testing.assert_allclose(first.matrix_encoded[:, income_index], expected_income)

            self.assertIn(None, first.raw_columns["region"].tolist())
            mode = first.categorical_fill_modes["region"]
            south_index = first.feature_indices.get("region=south")
            west_index = first.feature_indices.get("region=west")
            self.assertIsNotNone(south_index)
            self.assertIsNotNone(west_index)
            for row_index in first.train_indices:
                raw_value = first.raw_columns["region"][row_index]
                effective = mode if raw_value is None else raw_value
                self.assertEqual(first.matrix_encoded[row_index, south_index], 1.0 if effective == "south" else 0.0)
                self.assertEqual(first.matrix_encoded[row_index, west_index], 1.0 if effective == "west" else 0.0)

    def test_single_column_replacement_isolated_by_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            base = load_config(fixture.config_path)
            first = generate_dataset(base.data, base.seed)
            variant_path = fixture.variant_config("variant", column_variant=1)
            variant = load_config(variant_path)
            second = generate_dataset(variant.data, variant.seed)

            self.assertNotEqual(first.column_fingerprints["utilization"], second.column_fingerprints["utilization"])
            self.assertNotEqual(first.data_fingerprint, second.data_fingerprint)
            for column in ("income", "tenure", "bureau_score", "region", "channel", "income_copy"):
                self.assertEqual(first.column_fingerprints[column], second.column_fingerprints[column])
            self.assertEqual(first.target_fingerprint, second.target_fingerprint)
            self.assertEqual(first.split_fingerprint, second.split_fingerprint)
            self.assertEqual(first.train_sample_fingerprint, second.train_sample_fingerprint)

    def test_whitelist_order_and_model_order_do_not_change_semantic_config_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            original = load_config(fixture.config_path)
            fixture.config["models"] = list(reversed(fixture.config["models"]))
            for model in fixture.config["models"]:
                model["features"] = list(reversed(model["features"]))
            reordered_path = fixture.write_config("reordered.yaml")
            reordered = load_config(reordered_path)
            self.assertEqual(original.config_hash, reordered.config_hash)

            dataset = generate_dataset(original.data, original.seed)
            original_model = next(model for model in original.models if model.name == "elastic_balanced")
            reordered_model = next(model for model in reordered.models if model.name == "elastic_balanced")
            self.assertEqual(
                dataset.resolve_features(original_model.features),
                dataset.resolve_features(reordered_model.features),
            )

    def test_business_validation_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))

            fixture.config["models"][0]["scheme"] = "nuclear_norm"
            path = fixture.write_config("unknown-scheme.yaml")
            with self.assertRaises(LabError) as context:
                load_config(path)
            self.assertEqual(context.exception.code, E_SCHEME_UNKNOWN)

            fixture.config = copy.deepcopy(BASE_CONFIG)
            fixture.config["data"] = str(fixture.data_path)
            fixture.config["models"][2]["mix"] = 1.5
            path = fixture.write_config("bad-mix.yaml")
            with self.assertRaises(LabError) as context:
                load_config(path)
            self.assertEqual(context.exception.code, E_MIX_RANGE)

            fixture.config = copy.deepcopy(BASE_CONFIG)
            fixture.config["data"] = str(fixture.data_path)
            fixture.config["models"][2]["mix"] = float("nan")
            path = fixture.write_config("nan-mix.yaml")
            with self.assertRaises(LabError) as context:
                load_config(path)
            self.assertEqual(context.exception.code, E_MIX_RANGE)

            fixture.config = copy.deepcopy(BASE_CONFIG)
            fixture.config["data"] = str(fixture.data_path)
            fixture.config["models"][1]["features"] = ["not_a_column"]
            path = fixture.write_config("unknown-column.yaml")
            config = load_config(path)
            dataset = generate_dataset(config.data, config.seed)
            with self.assertRaises(LabError) as context:
                dataset.resolve_features(config.models[1].features)
            self.assertEqual(context.exception.code, E_CONFIG_INVALID)

            broken_data = yaml.safe_load(fixture.data_path.read_text(encoding="utf-8"))
            broken_data["columns"][0]["missing_rate"] = 1.2
            broken_path = Path(temporary) / "broken-data.yaml"
            broken_path.write_text(yaml.safe_dump(broken_data, sort_keys=False), encoding="utf-8")
            fixture.config = copy.deepcopy(BASE_CONFIG)
            fixture.config["data"] = str(broken_path)
            path = fixture.write_config("broken-data-config.yaml")
            config = load_config(path)
            with self.assertRaises(LabError) as context:
                generate_dataset(config.data, config.seed)
            self.assertEqual(context.exception.code, E_CONFIG_INVALID)


if __name__ == "__main__":
    unittest.main()
