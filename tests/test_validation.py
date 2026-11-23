from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from posdiag.models import (
    AttentionSample,
    ConfigValidationError,
    apply_edit_plan,
    load_samples,
    parse_config_dict,
)
from tests.helpers import BASE_CONFIG, make_config, make_sample


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_config_fields_emit_e_config_invalid(self) -> None:
        cases = [
            {"base_frequency": 0},
            {"training_context_length": 128, "extrapolation_target_length": 128},
            {"dimension": 7},
            {"batch_point_budget": 0},
            {"thread_pool_concurrency": 0},
            {"thread_pool_concurrency": (os.cpu_count() or 1) + 1},
            {"snapshot_commit_interval": 0},
            {"tolerance": 0},
            {"tolerance": float("inf")},
        ]
        for override in cases:
            with self.subTest(override=override):
                raw = dict(BASE_CONFIG)
                raw.update(override)
                with self.assertRaises(ConfigValidationError) as context:
                    parse_config_dict(raw)
                self.assertTrue(context.exception.issues)
                self.assertTrue(all(issue.code == "E_CONFIG_INVALID" for issue in context.exception.issues))

    def test_malformed_config_file_is_e_config_invalid(self) -> None:
        from posdiag.models import load_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(ConfigValidationError) as context:
                load_config(path)
            self.assertEqual(context.exception.issues[0].code, "E_CONFIG_INVALID")


class SampleValidationTests(unittest.TestCase):
    def test_invalid_lines_are_skipped_and_valid_samples_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            valid = {
                "sample_id": "s0001",
                "query_position": 2,
                "key_positions": [0, 1],
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "key_vectors": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
            }
            duplicate = dict(valid)
            negative = dict(valid, sample_id="s0003", query_position=-1)
            bad_dimension = dict(valid, sample_id="s0004", query_vector=[0.1, 0.2])
            non_finite = dict(valid, sample_id="s0005", query_vector=[0.1, float("nan"), 0.3, 0.4])
            lines = [
                json.dumps(valid),
                json.dumps(duplicate),
                json.dumps(negative),
                json.dumps(bad_dimension),
                json.dumps(non_finite),
                "{broken-json",
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            loaded = load_samples(path, dimension=4)
            self.assertEqual([sample.sample_id for sample in loaded.samples], ["s0001"])
            self.assertEqual(len(loaded.issues), 5)
            self.assertTrue(all(issue.code == "E_SAMPLE_INVALID" for issue in loaded.issues))
            self.assertEqual([issue.line for issue in loaded.issues], [2, 3, 4, 5, 6])

    def test_non_negative_positions_and_vector_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "bad",
                        "query_position": 1,
                        "key_positions": [1, -3],
                        "query_vector": [0, 0, 0, 0],
                        "key_vectors": [[0, 0, 0, 0], [0, 0, 0, 0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_samples(path, dimension=4)
            self.assertEqual(len(loaded.samples), 0)
            self.assertIn("key_positions[1]", loaded.issues[0].message)


class EditPlanTests(unittest.TestCase):
    def test_combined_edit_plan_changes_all_required_dimensions(self) -> None:
        config = make_config()
        samples = [make_sample("s0001"), make_sample("s0002")]
        replacement = make_sample("s0002", query_position=4, key_positions=(0, 1))
        plan = {
            "operations": [
                {"op": "switch_scheme", "scheme": "rope_ntk"},
                {"op": "adjust_base_frequency", "factor": 1.5},
                {"op": "change_precision", "precision": "compact"},
                {"op": "replace_sample", "sample_id": "s0002", "sample": replacement.content_dict()},
            ]
        }
        edited = apply_edit_plan(config, samples, plan)
        self.assertEqual(edited.issues, ())
        self.assertEqual(edited.config.scheme, "rope_ntk")
        self.assertAlmostEqual(edited.config.base_frequency, 15000.0)
        self.assertEqual(edited.config.precision, "compact")
        self.assertEqual(edited.samples[1].query_position, 4)

    def test_invalid_edit_does_not_prevent_later_valid_edit(self) -> None:
        config = make_config()
        plan = {
            "operations": [
                {"op": "unknown"},
                {"op": "change_precision", "precision": "balanced"},
            ]
        }
        edited = apply_edit_plan(config, [make_sample("s0001")], plan)
        self.assertEqual(edited.config.precision, "balanced")
        self.assertEqual(len(edited.issues), 1)
        self.assertEqual(edited.issues[0].code, "E_EDIT_INVALID")


if __name__ == "__main__":
    unittest.main()
