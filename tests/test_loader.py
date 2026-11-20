from __future__ import annotations

import unittest

from rope_extrap.loader import load_position_config, load_samples
from tests.helpers import CONFIG_LARGE, SAMPLES_DIR


class LoaderTests(unittest.TestCase):
    def test_fault_samples_are_isolated_with_expected_codes(self) -> None:
        config, failures = load_position_config(CONFIG_LARGE)
        self.assertIsNotNone(config)
        self.assertEqual(failures, [])
        assert config is not None
        samples, sample_failures = load_samples(
            SAMPLES_DIR / "fault_samples.jsonl",
            dimension=config.dimension,
            target_length=config.extrapolation_target_length,
        )
        self.assertEqual(sorted(samples), ["f001", "f007"])
        codes = [failure.code for failure in sample_failures]
        self.assertIn("E_SAMPLE_INVALID", codes)
        self.assertIn("E_DIM_MISMATCH", codes)
        self.assertEqual(len([code for code in codes if code == "E_SAMPLE_INVALID"]), 4)

    def test_empty_file_and_invalid_json(self) -> None:
        _samples, failures = load_samples(
            SAMPLES_DIR / "empty_samples.jsonl",
            dimension=32,
            target_length=128,
        )
        self.assertEqual([failure.code for failure in failures], ["E_SAMPLE_INVALID"])
        config, failures = load_position_config(SAMPLES_DIR / "config_invalid_json.json")
        self.assertIsNone(config)
        self.assertEqual([failure.code for failure in failures], ["E_CONFIG_INVALID"])

    def test_base_and_scale_failures_have_specific_codes(self) -> None:
        config, failures = load_position_config(SAMPLES_DIR / "config_fault_base.json")
        self.assertIsNone(config)
        self.assertEqual([failure.code for failure in failures], ["E_BASE_INVALID"])
        config, failures = load_position_config(SAMPLES_DIR / "config_fault_scale.json")
        self.assertIsNone(config)
        self.assertEqual([failure.code for failure in failures], ["E_SCALE_INVALID"])


if __name__ == "__main__":
    unittest.main()

