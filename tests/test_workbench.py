from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import copy
from pathlib import Path

from evalworkbench.closure import build_context, compute_frame_closure, compute_segment_closure
from evalworkbench.constants import DEFAULT_INJECTION, ENCODING_PARAMS, METRIC_POLICY
from evalworkbench.errors import BusinessError
from evalworkbench.formats import parse_metric_binary, parse_project_json
from evalworkbench.generator import generate_assets
from evalworkbench.ledger import LedgerStore
from evalworkbench.pipeline import (
    build_carriers,
    evaluate_workspace,
    load_truth,
    preview_workspace,
    recover_workspace,
)
from evalworkbench.quality import validate_shots


ROOT = Path(__file__).resolve().parents[1]
INJECTIONS = ROOT / "injections"
MUTATIONS = ROOT / "mutations"


class WorkbenchAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="evalworkbench-test-")
        cls.workspace = Path(cls.tmp.name)
        cls.generation = generate_assets(cls.workspace)
        cls.carriers = build_carriers(cls.workspace)
        cls.baseline = evaluate_workspace(cls.workspace)
        if cls.baseline["status"] != "ok":
            raise AssertionError(cls.baseline["business_errors"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _latest(self):
        snapshot = LedgerStore(self.workspace).load_latest_committed()
        self.assertIsNotNone(snapshot)
        return snapshot

    def _evaluate(self, injection: str | None = None):
        injection_path = INJECTIONS / injection if injection else None
        return evaluate_workspace(self.workspace, injection_path=injection_path)

    def test_01_fixture_contract_and_counts(self) -> None:
        self.assertEqual(self.generation["frame_count"], 300)
        self.assertEqual(self.generation["shot_count"], 4)
        self.assertEqual(self.generation["audio_sample_count_per_channel"], 480_000)
        self.assertEqual(self.generation["audio_total_samples"], 960_000)
        truth = load_truth(self.workspace)
        self.assertEqual(truth["video"]["width"], 640)
        self.assertEqual(truth["video"]["height"], 360)
        self.assertEqual(truth["video"]["fps"], 30)
        self.assertEqual(truth["audio"]["sample_rate"], 48_000)
        self.assertEqual(truth["shots_boundaries"], [0, 90, 180, 240])
        self.assertTrue(Path(self.workspace / "assets" / "source.y4m").exists())
        self.assertTrue(Path(self.workspace / "assets" / "encoded.y4m").exists())
        self.assertTrue(Path(self.workspace / "assets" / "audio.wav").exists())

    def test_02_baseline_equivalence_and_byte_stable_metrics(self) -> None:
        first = self._latest()
        report_digest = self.baseline["equivalence"]["report_digest"]
        second = self._evaluate()
        self.assertEqual(second["status"], "ok")
        self.assertEqual(second["closure"]["reason"], "no_changes")
        latest = self._latest()
        for shard in ("actual_frames.json", "actual_segments.json", "audio.json", "equivalence.json"):
            self.assertEqual(first["manifest"]["shard_hashes"][shard], latest["manifest"]["shard_hashes"][shard])
        self.assertEqual(report_digest, second["equivalence"]["report_digest"])
        statuses = {(pair["left"], pair["right"]): pair["status"] for pair in second["equivalence"]["pairs"]}
        self.assertEqual(statuses[("project_json", "metric_binary")], "equivalent")
        self.assertEqual(statuses[("project_json", "legacy_json")], "equivalent_with_assumptions")
        self.assertEqual(statuses[("metric_binary", "legacy_json")], "equivalent_with_assumptions")
        self.assertTrue(all(claim["status"] == "equivalent" for claim in second["carrier_claims"]))

    def test_03_metric_version_migration_is_idempotent(self) -> None:
        from evalworkbench.formats import migrate_metric_record

        v1 = parse_metric_binary(self.workspace / "carriers" / "metrics_v1.emf")
        v2 = parse_metric_binary(self.workspace / "carriers" / "metrics_v2.emf")
        self.assertEqual(v1.provenance["metric_encoding_version"], 1)
        for before, after in zip(v1.declared_frames, v2.declared_frames, strict=True):
            for field in ("frame_index", "shot_id", "timestamp_ms", "psnr_db", "ssim", "flags"):
                self.assertEqual(before[field], after[field])
        migrated, changed = migrate_metric_record(v1.declared_frames[0], 1)
        migrated_again, changed_again = migrate_metric_record(migrated, 2)
        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertEqual(migrated, migrated_again)
        self.assertIsNone(migrated["chroma_psnr_db"])

    def test_04_parser_business_codes_are_isolated(self) -> None:
        cases = {
            "binary_field_bitmap.json": "E_FIELD_BITMAP",
            "binary_metric_version.json": "E_METRIC_VERSION",
            "binary_endian.json": "E_ENDIAN_MISMATCH",
            "binary_index_width.json": "E_INDEX_OVERFLOW",
        }
        for filename, expected_code in cases.items():
            with self.subTest(filename=filename):
                result = self._evaluate(filename)
                codes = {error["code"] for error in result["business_errors"]}
                self.assertIn(expected_code, codes)
                self.assertIn("project_json", {claim["carrier"] for claim in result["carrier_claims"]})
                binary_pairs = [
                    pair
                    for pair in result["equivalence"]["pairs"]
                    if "metric_binary" in (pair["left"], pair["right"])
                ]
                self.assertTrue(all(pair["status"] == "invalid_carrier" for pair in binary_pairs))
                self.assertFalse(result["equivalence"]["overall"]["equivalent"])
        bad_schema = self.workspace / "carriers" / "bad_schema.json"
        document = json.loads((self.workspace / "carriers" / "project_v2.json").read_text(encoding="utf-8"))
        document["schema_version"] = "eval-project/9.9"
        bad_schema.write_text(json.dumps(document), encoding="utf-8")
        try:
            with self.assertRaises(BusinessError) as context:
                parse_project_json(bad_schema)
            self.assertEqual(context.exception.code, "E_SCHEMA_VERSION")
        finally:
            bad_schema.unlink(missing_ok=True)

    def test_05_audio_and_frame_fault_detection(self) -> None:
        drift = self._evaluate("audio_drift.json")
        self.assertEqual(drift["audio"]["overall"]["status"], "drift")
        self.assertEqual(drift["audio"]["overall"]["first_drift_segment"], 0)
        self.assertAlmostEqual(drift["audio"]["segments"][0]["offset_samples"], 96, places=6)
        self.assertAlmostEqual(drift["audio"]["segments"][0]["gain_ratio"], 1.05, places=4)
        self.assertIn("E_AUDIO_DRIFT", {error["code"] for error in drift["business_errors"]})

        rate_lie = self._evaluate("audio_rate_lie.json")
        self.assertEqual(rate_lie["audio"]["error"]["code"], "E_AUDIO_SAMPLE_RATE")

        faults = self._evaluate("drop_and_timestamp.json")
        self.assertEqual(faults["missing_units"]["missing_frames"], [7, 150])
        codes = {error["code"] for error in faults["business_errors"]}
        self.assertIn("E_DROPPED_FRAME", codes)
        self.assertIn("E_TIMESTAMP_JUMP", codes)
        self.assertIn("E_EQUIV_MISMATCH", codes)
        self.assertTrue(all(claim["status"] == "mismatch" for claim in faults["carrier_claims"]))

        closure = self._evaluate("closure_mismatch.json")
        self.assertFalse(closure["closure"]["match"])
        self.assertIn("E_CLOSURE_MISMATCH", {error["code"] for error in closure["business_errors"]})

    def test_05b_encoding_policy_and_shot_closures_are_exact(self) -> None:
        reset = self._evaluate()
        self.assertEqual(reset["status"], "ok")
        snapshot = self._latest()
        prior_frames = snapshot["shards"]["actual_frames.json"]["records"]
        prior_segments = snapshot["shards"]["actual_segments.json"]["segments"]
        content_hash = snapshot["manifest"]["content_hash"]
        shots = [dict(shot) for shot in load_truth(self.workspace)["shots"]]
        injection = copy.deepcopy(DEFAULT_INJECTION)

        base_context = build_context(content_hash, ENCODING_PARAMS, METRIC_POLICY, shots, injection, 300)
        self.assertEqual(compute_frame_closure(prior_frames, base_context), [])
        self.assertEqual(compute_segment_closure(prior_segments, shots, prior_frames, base_context), [])

        encoding = copy.deepcopy(ENCODING_PARAMS)
        encoding["qstep"] = 9
        encoding_context = build_context(content_hash, encoding, METRIC_POLICY, shots, injection, 300)
        self.assertEqual(compute_frame_closure(prior_frames, encoding_context), list(range(300)))
        self.assertEqual(
            compute_segment_closure(prior_segments, shots, prior_frames, encoding_context),
            [0, 1, 2, 3],
        )

        policy = copy.deepcopy(METRIC_POLICY)
        policy["ssim_window"] = 4
        policy_context = build_context(content_hash, ENCODING_PARAMS, policy, shots, injection, 300)
        self.assertEqual(compute_frame_closure(prior_frames, policy_context), list(range(300)))
        self.assertEqual(
            compute_segment_closure(prior_segments, shots, prior_frames, policy_context),
            [0, 1, 2, 3],
        )

        merged_shots = validate_shots(json.loads((MUTATIONS / "shots_merge_1_2.json").read_text())["shots"], 300)
        shot_context = build_context(content_hash, ENCODING_PARAMS, METRIC_POLICY, merged_shots, injection, 300)
        self.assertEqual(compute_frame_closure(prior_frames, shot_context), list(range(180, 300)))
        self.assertEqual(
            compute_segment_closure(prior_segments, merged_shots, prior_frames, shot_context),
            [1, 2, 3],
        )

    def test_06_precise_shot_closure_preserves_other_hashes(self) -> None:
        reset = self._evaluate()
        self.assertEqual(reset["status"], "ok")
        before = self._latest()
        before_frames = {
            int(record["frame_index"]): record
            for record in before["shards"]["actual_frames.json"]["records"]
        }
        result = evaluate_workspace(self.workspace, mutation_path=MUTATIONS / "shots_merge_1_2.json")
        self.assertEqual(result["closure"]["changed_inputs"], ["shots"])
        self.assertEqual(result["closure"]["frames"], list(range(180, 300)))
        self.assertEqual(result["closure"]["segments"], [1, 2, 3])
        self.assertTrue(result["closure"]["match"])
        self.assertEqual(
            result["closure"]["actual_recomputed_frames"],
            result["closure"]["frames"],
        )
        after = self._latest()
        after_frames = {
            int(record["frame_index"]): record
            for record in after["shards"]["actual_frames.json"]["records"]
        }
        changed = {
            frame_index
            for frame_index, record in before_frames.items()
            if record["record_hash"] != after_frames[frame_index]["record_hash"]
        }
        self.assertEqual(changed, set(result["closure"]["frames"]))

    def test_07_crash_recovery_uses_last_committed_snapshot(self) -> None:
        baseline = self._evaluate()
        self.assertEqual(baseline["status"], "ok")
        previous = self._latest()
        command = [
            sys.executable,
            "-m",
            "evalworkbench.cli",
            "evaluate",
            "--workspace",
            str(self.workspace),
            "--crash-point",
            "before_manifest",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 92, completed.stderr)
        recovery = recover_workspace(self.workspace)
        self.assertTrue(recovery["recovered"])
        self.assertEqual(recovery["latest_revision"], previous["manifest"]["revision"])
        self.assertFalse(recovery["normal_exit_confirmed"])
        self.assertGreaterEqual(recovery["orphan_count"], 1)
        self.assertFalse((self.workspace / "ledger" / "closed.json").exists())

    def test_08_stale_snapshot_is_rejected(self) -> None:
        latest = self._latest()
        with self.assertRaises(BusinessError) as context:
            evaluate_workspace(self.workspace, expected_revision=latest["manifest"]["revision"] + 100)
        self.assertEqual(context.exception.code, "E_SNAPSHOT_STALE")

    def test_09_decode_preview(self) -> None:
        output = self.workspace / "preview_frame_000.ppm"
        preview = preview_workspace(self.workspace, 0, output)
        self.assertEqual(preview["width"], 640)
        self.assertEqual(preview["height"], 360)
        self.assertTrue(output.read_bytes().startswith(b"P6\n640 360\n255\n"))


if __name__ == "__main__":
    unittest.main()
