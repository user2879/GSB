from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from timbre_workbench.assign import fold_assignment, pair_decision
from timbre_workbench.config import load_config, load_labels
from timbre_workbench.engine import Workbench, semantic_projection
from timbre_workbench.synth import build_file, write_minimal_scenario
from timbre_workbench.util import canonical_json
from timbre_workbench.wavio import write_wav


class FeatureQuantizationTests(unittest.TestCase):
    def test_features_are_deterministic_and_formants_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scenario = write_minimal_scenario(Path(temporary), invalid_audio=True)
            config = load_config(Path(scenario["config"]), overlay={"runtime": {"first_response_budget_ms": 25}})
            labels = load_labels(Path(scenario["labels"]))
            workbench = Workbench.from_directory(
                Path(scenario["audio_dir"]), config=config, labels=labels, snapshot_manager=None
            )
            workbench.run_commands([])
            report_a = workbench.refresh_report()
            digest_a = report_a["canonical_digest"]
            report_b = workbench.rebuild_full(commit=False)
            self.assertEqual(digest_a, report_b["canonical_digest"])
            self.assertEqual(canonical_json(semantic_projection(report_a)), canonical_json(semantic_projection(report_b)))
            self.assertGreaterEqual(len(report_a["segments"]), 3)
            for segment in report_a["segments"]:
                self.assertGreater(segment["features"]["valid_frame_count"], 0)
                self.assertGreater(segment["features"]["f1_mean_hz"], 300.0)
                self.assertGreater(segment["features"]["f2_mean_hz"], segment["features"]["f1_mean_hz"])
                self.assertGreater(segment["features"]["f3_mean_hz"], segment["features"]["f2_mean_hz"])
                self.assertEqual(segment["features"]["accumulation_order"], "frame_index_ascending")
                self.assertEqual(
                    segment["features"]["frame_indices"], sorted(segment["features"]["frame_indices"])
                )
            self.assertEqual(report_a["failures"][0]["error_code"], "E_AUDIO_FORMAT")

    def test_tier_switch_changes_integer_codes_and_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scenario = write_minimal_scenario(Path(temporary))
            config = load_config(Path(scenario["config"]), overlay={"runtime": {"first_response_budget_ms": 25}})
            labels = load_labels(Path(scenario["labels"]))
            reports = {}
            for tier in (4, 8):
                workbench = Workbench.from_directory(
                    Path(scenario["audio_dir"]), config=config, labels=labels, snapshot_manager=None
                )
                workbench.run_commands(
                    [{"op": "set_quantization_tier", "dimension": "f0_mean", "tier": tier}],
                    command_base_dir=Path(temporary),
                )
                reports[tier] = workbench.refresh_report()
            low = reports[4]["segments"]
            high = reports[8]["segments"]
            changed_codes = False
            changed_vectors = False
            for left, right in zip(low, high):
                left_dim = left["description"]["dimension_records"][0]
                right_dim = right["description"]["dimension_records"][0]
                self.assertEqual(left_dim["dimension"], "f0_mean")
                self.assertEqual(left_dim["tier_bits"], 4)
                self.assertEqual(right_dim["tier_bits"], 8)
                changed_codes = changed_codes or left_dim["quantized_integer"] != right_dim["quantized_integer"]
                changed_vectors = changed_vectors or left["description"]["vector"] != right["description"]["vector"]
            self.assertTrue(changed_codes)
            self.assertTrue(changed_vectors)


class AssignmentTests(unittest.TestCase):
    def test_low_threshold_forces_conflicts_but_keeps_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scenario = write_minimal_scenario(Path(temporary))
            config = load_config(Path(scenario["config"]), overlay={"runtime": {"first_response_budget_ms": 25}})
            labels = load_labels(Path(scenario["labels"]))
            workbench = Workbench.from_directory(
                Path(scenario["audio_dir"]), config=config, labels=labels, snapshot_manager=None
            )
            workbench.run_commands(
                [{"op": "set_similarity_threshold", "value": 0.0}],
                command_base_dir=Path(temporary),
            )
            report = workbench.refresh_report()
            self.assertEqual(len(report["speakers"]), 1)
            self.assertEqual(len(report["conflicts"]), 2)
            self.assertTrue(all(row["resolution"] == "keep_prediction_record_conflict" for row in report["conflicts"]))
            speaker = report["speakers"][0]
            self.assertFalse(speaker["aggregation_degenerate"])
            self.assertEqual(len(speaker["participants"]), 3)
            self.assertEqual(
                speaker["weight_denominator"],
                sum(row["features"]["valid_frame_count"] for row in report["segments"]),
            )

    def test_single_segment_speaker_is_marked_degenerate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            audio.mkdir()
            samples, labels_data = build_file(16000, [("alpha", 300.0, 77)], gap_ms=180.0)
            write_wav(audio / "single.wav", samples)
            config = load_config(overlay={"runtime": {"first_response_budget_ms": 25}})
            labels_path = root / "labels.json"
            labels_path.write_text(
                json.dumps({"files": {"single.wav": labels_data}}), encoding="utf-8"
            )
            workbench = Workbench.from_directory(
                audio, config=config, labels=load_labels(labels_path), snapshot_manager=None
            )
            workbench.run_commands([], command_base_dir=root)
            report = workbench.refresh_report()
            self.assertEqual(len(report["segments"]), 1)
            self.assertTrue(report["speakers"][0]["aggregation_degenerate"])

    def test_exact_threshold_uses_speaker_id_lexicographic_tie_break(self) -> None:
        decision = pair_decision(0.5, 0.5)
        self.assertTrue(decision["tie_at_threshold"])
        rows = [
            {
                "file": "x.wav",
                "segment_id": "s1",
                "segment_index": 0,
                "annotation": "spk_a",
                "pair_decision_key": None,
            },
            {
                "file": "x.wav",
                "segment_id": "s2",
                "segment_index": 1,
                "annotation": "spk_b",
                "pair_decision_key": "pair",
            },
        ]
        assignments, conflicts = fold_assignment(rows, {"pair": decision}, load_config()["assignment"])
        self.assertEqual(assignments[0]["speaker_id"], "spk_a")
        self.assertEqual(assignments[1]["speaker_id"], "spk_a")
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(conflicts[0]["predicted_same"])
        self.assertFalse(conflicts[0]["annotated_same"])

        rows[0]["annotation"] = "spk_b"
        rows[1]["annotation"] = "spk_a"
        assignments, conflicts = fold_assignment(rows, {"pair": decision}, load_config()["assignment"])
        self.assertNotEqual(assignments[1]["speaker_id"], assignments[0]["speaker_id"])
        self.assertEqual(conflicts, [])
        self.assertEqual(assignments[1]["decision_reason"], "threshold_equality_speaker_id_lexicographic")


if __name__ == "__main__":
    unittest.main()
