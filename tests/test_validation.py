from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rope_diag.config import SCHEMES, load_config, validate_config_dict
from rope_diag.errors import ErrorCode
from rope_diag.injections import load_injections
from rope_diag.pruning import load_pruning_policies
from rope_diag.samples import load_samples

from tests.support import config_dict, injection_dict, pruning_dict, write_case


class ValidationTests(unittest.TestCase):
    def test_config_accepts_all_three_schemes_and_rejects_bad_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root)
            config, failures, _ = load_config(paths["config"])
            self.assertIsNotNone(config)
            self.assertEqual(failures, [])
            self.assertEqual(config.schemes, SCHEMES)

            invalid = config_dict(
                base=0,
                dimension=3,
                training_context_length=16,
                extrapolation_target_length=16,
                scaling_factor=0,
                precision="fp16",
                quantization="fp8",
                point_budget=0,
                concurrency=0,
            )
            _, invalid_failures = validate_config_dict(
                invalid, source_path=Path("invalid.json"), config_hash=""
            )
            fields = {
                failure.details.get("field")
                for failure in invalid_failures
                if failure.code == ErrorCode.CONFIG_INVALID
            }
            self.assertTrue(
                {
                    "base",
                    "dimension",
                    "extrapolation_target_length",
                    "scaling_factor",
                    "precision",
                    "quantization",
                    "point_budget",
                    "concurrency",
                }.issubset(fields)
            )

    def test_sample_jsonl_validation_duplicate_ids_and_sorted_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = {
                "sample_id": "s0002",
                "query_position": 2,
                "key_positions": [0, 1],
                "query_features": [0.0] * 8,
                "key_features": [[0.0] * 8, [0.0] * 8],
            }
            duplicate = dict(good)
            negative = dict(good)
            negative["sample_id"] = "s0001"
            negative["query_position"] = -1
            duplicate_keys = dict(good)
            duplicate_keys["sample_id"] = "s0003"
            duplicate_keys["key_positions"] = [1, 1]
            nonfinite = dict(good)
            nonfinite["sample_id"] = "s0004"
            nonfinite["query_features"] = [float("nan")] + [0.0] * 7
            path = root / "samples.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in (good, duplicate, negative, duplicate_keys, nonfinite):
                    handle.write(json.dumps(row) + "\n")
                handle.write("{not json}\n")
            dataset = load_samples(path, dimension=8)
            self.assertEqual(dataset.sample_count, 1)
            self.assertGreaterEqual(len(dataset.failures), 5)
            sample_ids = [failure.sample_id for failure in dataset.failures]
            self.assertEqual(sample_ids, sorted(sample_ids))
            self.assertTrue(all(failure.code == ErrorCode.SAMPLE_INVALID for failure in dataset.failures))

    def test_pruning_and_injection_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pruning_path = root / "pruning.json"
            pruning_path.write_text(json.dumps(pruning_dict()), encoding="utf-8")
            policies, failures, _ = load_pruning_policies(pruning_path)
            self.assertEqual(len(policies), 4)
            self.assertEqual(failures, [])

            invalid_pruning = {
                "policies": [
                    {
                        "policy_id": "bad",
                        "pair_ranges": [{"start_ratio": 0.9, "end_ratio": 0.1}],
                        "pair_stride": 0,
                    }
                ]
            }
            pruning_path.write_text(json.dumps(invalid_pruning), encoding="utf-8")
            _, invalid_policy_failures, _ = load_pruning_policies(pruning_path)
            self.assertTrue(
                all(failure.code == ErrorCode.CONFIG_INVALID for failure in invalid_policy_failures)
            )

            injection_path = root / "injections.json"
            injection_path.write_text(
                json.dumps(injection_dict([
                    {
                        "injection_id": "INJ-X",
                        "scheme": "linear_interpolation",
                        "sample_id": "s0001",
                        "stage": "scoring",
                        "action": "raise",
                    }
                ])),
                encoding="utf-8",
            )
            injections, injection_failures, _ = load_injections(injection_path)
            self.assertEqual(len(injections), 1)
            self.assertEqual(injection_failures, [])

            injection_path.write_text(
                json.dumps({"injections": [{"injection_id": "bad", "stage": "nope", "action": "explode"}]}),
                encoding="utf-8",
            )
            _, bad_injection_failures, _ = load_injections(injection_path)
            self.assertTrue(
                all(failure.code == ErrorCode.CONFIG_INVALID for failure in bad_injection_failures)
            )


if __name__ == "__main__":
    unittest.main()
