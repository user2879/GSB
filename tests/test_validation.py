from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from ctr_cvr_pipeline.config import ConfigError, feature_config_from_dict, training_config_from_dict
from ctr_cvr_pipeline.samples import load_samples_jsonl

from tests.helpers import feature_config_dict, make_raw, training_config_dict


class ValidationTests(unittest.TestCase):
    def test_invalid_sample_collects_sample_id_and_field(self) -> None:
        config = feature_config_from_dict(feature_config_dict())
        bad = make_raw(1, 0)
        bad["dense"]["price"] = -2
        bad["click"] = 3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            samples, failures = load_samples_jsonl(path, config, event_time_ms=0)
        self.assertEqual(samples, [])
        codes = {failure.code for failure in failures}
        self.assertEqual(codes, {"E_SAMPLE_INVALID"})
        fields = {failure.field for failure in failures}
        self.assertIn("dense.price", fields)
        self.assertIn("click", fields)
        self.assertTrue(all(failure.sample_id == "R0001:I0001_0:line1" for failure in failures))

    def test_duplicate_item_in_request_is_invalid(self) -> None:
        config = feature_config_from_dict(feature_config_dict())
        first = make_raw(1, 0)
        duplicate = dict(first)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(duplicate) + "\n",
                encoding="utf-8",
            )
            samples, failures = load_samples_jsonl(path, config, event_time_ms=0)
        self.assertEqual(len(samples), 1)
        self.assertTrue(any(failure.field == "item_id" for failure in failures))

    def test_config_rejects_unknown_shared_feature(self) -> None:
        raw = feature_config_dict()
        raw["shared_features"] = ["does_not_exist"]
        with self.assertRaises(ConfigError):
            feature_config_from_dict(raw)

    def test_config_rejects_non_positive_weights(self) -> None:
        raw = training_config_dict(negative_weight=0)
        with self.assertRaises(ConfigError):
            training_config_from_dict(raw)

    def test_config_rejects_incompatible_quantization(self) -> None:
        raw = feature_config_dict(tier="Q8", bits=8)
        with self.assertRaises(ConfigError):
            feature_config_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
