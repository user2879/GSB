from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SAMPLES, canonical_files, engine_paths, sample_config, write_json_file
from voicecal.constants import E_CALIB_BUDGET_PENDING, E_CONFIG_MISMATCH
from voicecal.engine import run_engine
from voicecal.errors import VoicecalError
from voicecal.util import file_hash, read_json, write_json


class EngineIntegrationTests(unittest.TestCase):
    def test_budgeted_incremental_equals_full_and_resumes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=2, concurrency=3)
            reports = []
            iterations = 0
            while True:
                iterations += 1
                report = run_engine(engine_paths(tmp, config=config, name="incremental"), fresh=False)
                reports.append(report)
                if report["state"]["complete"]:
                    break
                self.assertIn(E_CALIB_BUDGET_PENDING, report["business_codes"])
                self.assertTrue(report["pending"])
                if iterations > 20:
                    self.fail("incremental run did not converge")
            self.assertGreater(iterations, 1)
            self.assertTrue(reports[0].get("snapshot_resumed") is False)
            self.assertTrue(any(report.get("snapshot_resumed") for report in reports[1:]))
            self.assertTrue(any(sum(report["ledger"]["hit_counts"].values()) > 0 for report in reports[1:]))

            snapshot_root = tmp / "incremental" / "snapshot"
            manifest = read_json(snapshot_root / "manifest.json")
            for stage in ("extract", "initial", "calibration", "attribution"):
                metadata = manifest["stages"][stage]
                self.assertEqual(metadata["stage"], stage)
                self.assertIn("config_hash", metadata)
                self.assertIn("injection_hash", metadata)
                self.assertIn("turn_cursor", metadata)
                self.assertEqual(metadata["fixed_point_bits"], 32)
                self.assertIn("artifact_version", metadata)

            full_report = run_engine(engine_paths(tmp, config=config, name="full"), fresh=True)
            self.assertTrue(full_report["state"]["complete"])
            incremental_output = tmp / "incremental" / "output"
            full_output = tmp / "full" / "output"
            for filename in canonical_files():
                self.assertEqual(
                    file_hash(incremental_output / filename),
                    file_hash(full_output / filename),
                    f"canonical artifact differs: {filename}",
                )

            ledger = read_json(incremental_output / "calibration_ledger.json")
            self.assertEqual([entry["turn_index"] for entry in ledger["entries"]], [1, 2, 3, 4, 5, 6, 8, 9, 10])
            self.assertTrue(all(entry["cache_keys"]["calibration"] for entry in ledger["entries"]))
            attribution = read_json(incremental_output / "attribution.json")
            self.assertGreater(len(attribution["trajectories"]), 0)
            self.assertTrue(all(trajectory["turn_ids"] for trajectory in attribution["trajectories"]))
            effective = read_json(incremental_output / "effective_dialogue.json")
            by_id = {turn["turn_id"]: turn for turn in effective["turns"]}
            self.assertEqual(by_id["t0003"]["audio"], "replacements/t0003_v2.wav")
            self.assertEqual(by_id["t0006"]["speaker_id"], "spk_b")
            self.assertNotIn("t0007", by_id)
            self.assertIn("t0010", by_id)

    def test_snapshot_version_mismatch_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=5)
            run_engine(engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"), fresh=True)
            manifest_path = tmp / "run" / "snapshot" / "manifest.json"
            manifest = read_json(manifest_path)
            original = json.loads(json.dumps(manifest))
            manifest["artifact_version"] = "incompatible-snapshot"
            write_json(manifest_path, manifest)
            with self.assertRaises(VoicecalError) as context:
                run_engine(engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"), fresh=False)
            self.assertEqual(context.exception.code, E_CONFIG_MISMATCH)
            original["base_config_hash"] = "0" * 64
            write_json(manifest_path, original)
            with self.assertRaises(VoicecalError) as context:
                run_engine(engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"), fresh=False)
            self.assertEqual(context.exception.code, E_CONFIG_MISMATCH)

    def test_gain_edit_reuses_extraction_and_invalidates_calibration_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=20)
            run_engine(
                engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"),
                fresh=True,
            )
            gain_edit = write_json_file(
                tmp / "gain_edit.json",
                {
                    "schema_version": 1,
                    "batches": [
                        {
                            "batch_id": "gain-only",
                            "commands": [{"op": "adjust_gain", "turn_id": "t0005", "gain": 0.61}],
                        }
                    ],
                },
            )
            report = run_engine(engine_paths(tmp, config=config, edits=gain_edit), fresh=False)
            events = report["ledger"]["events"]
            extract_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "extract"}
            initial_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "initial"}
            calibration_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "calibration"}
            self.assertTrue(all(status == "reused" for status in extract_events.values()))
            self.assertTrue(all(status == "reused" for status in initial_events.values()))
            for turn_id in ("t0001", "t0002", "t0003", "t0004"):
                self.assertEqual(calibration_events[turn_id], "reused")
            for turn_id in ("t0005",):
                self.assertEqual(calibration_events[turn_id], "recomputed")
            for turn_id in ("t0006", "t0007", "t0008", "t0009"):
                self.assertEqual(calibration_events[turn_id], "reused")
            self.assertEqual(report["ledger"]["invalidation_counts"]["extract"], 0)
            self.assertEqual(report["ledger"]["invalidation_counts"]["calibration"], 1)

    def test_audio_replace_invalidates_only_that_turn_and_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=20)
            run_engine(
                engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"),
                fresh=True,
            )
            replace_edit = write_json_file(
                tmp / "replace_edit.json",
                {
                    "schema_version": 1,
                    "batches": [
                        {
                            "batch_id": "replace-only",
                            "commands": [
                                {"op": "replace_audio", "turn_id": "t0003", "audio": "replacements/t0003_v2.wav"}
                            ],
                        }
                    ],
                },
            )
            report = run_engine(engine_paths(tmp, config=config, edits=replace_edit), fresh=False)
            events = report["ledger"]["events"]
            extract_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "extract"}
            initial_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "initial"}
            calibration_events = {event["turn_id"]: event["status"] for event in events if event["stage"] == "calibration"}
            self.assertEqual(extract_events["t0003"], "recomputed")
            self.assertEqual(initial_events["t0003"], "recomputed")
            for turn_id in ("t0001", "t0002", "t0004", "t0005", "t0006", "t0007", "t0008", "t0009"):
                self.assertEqual(extract_events[turn_id], "reused")
                self.assertEqual(initial_events[turn_id], "reused")
            for turn_id in ("t0003", "t0004", "t0005"):
                self.assertEqual(calibration_events[turn_id], "recomputed")
            for turn_id in ("t0006", "t0007", "t0008", "t0009"):
                self.assertEqual(calibration_events[turn_id], "reused")
            self.assertEqual(calibration_events["t0001"], "reused")
            self.assertEqual(calibration_events["t0002"], "reused")

    def test_injections_hit_and_failed_turns_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=20)
            report = run_engine(
                engine_paths(
                    tmp,
                    config=config,
                    edits=SAMPLES / "no_edits.json",
                    injections=SAMPLES / "injections_validation.json",
                ),
                fresh=True,
            )
            hit_codes = {(hit["injection_id"], hit["business_code"]) for hit in report["injection"]["hits"]}
            self.assertIn(("INJ-AUDIO-001", "E_AUDIO_FORMAT"), hit_codes)
            self.assertIn(("INJ-CAL-001", "E_INJECTED_FAULT"), hit_codes)
            failed_turns = {failure["turn_id"] for failure in report["failures"]}
            self.assertIn("t0004", failed_turns)
            self.assertIn("t0006", failed_turns)
            attribution = read_json(tmp / "run" / "output" / "attribution.json")
            attributed_turns = {
                item["turn_id"]
                for item in attribution["turns"]
                if item["record"] is not None and item["record"].get("status") == "ok"
            }
            self.assertNotIn("t0004", attributed_turns)
            self.assertNotIn("t0006", attributed_turns)

    def test_short_segments_fail_attribution_but_calibration_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=20, min_segment_ms=99999)
            report = run_engine(
                engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"),
                fresh=True,
            )
            self.assertIn("E_SEGMENT_TOO_SHORT", report["business_codes"])
            attribution = read_json(tmp / "run" / "output" / "attribution.json")
            self.assertEqual(attribution["trajectories"], [])
            self.assertTrue(all(item["status"] == "failed" for item in attribution["turns"]))
            calibration = read_json(tmp / "run" / "output" / "calibration.json")
            self.assertTrue(all(item["status"] == "ok" for item in calibration["turns"]))

    def test_concurrency_limit_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config = sample_config(tmp, budget=20, concurrency=2)
            report = run_engine(
                engine_paths(tmp, config=config, edits=SAMPLES / "no_edits.json"),
                fresh=True,
            )
            self.assertEqual(report["concurrency"]["configured_max_workers"], 2)
            self.assertLessEqual(report["concurrency"]["max_active_frame_workers_observed"], 2)
            self.assertGreater(report["work"]["frame_tasks"], 0)
            self.assertGreater(report["work"]["aggregation_tasks"], 0)
            self.assertGreater(report["work"]["initial_evidence_tasks"], 0)
            self.assertLessEqual(report["work"]["max_active_initial_workers_observed"], 2)


if __name__ == "__main__":
    unittest.main()
